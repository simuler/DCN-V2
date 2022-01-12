# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math


class DeepCroLayer(nn.Layer):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes, cross_num,
                 clip_by_norm, l2_reg_cross, is_sparse, num_experts, low_rank, struct, net_struct):
        super(DeepCroLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes
        self.cross_num = cross_num
        self.clip_by_norm = clip_by_norm
        self.l2_reg_cross = l2_reg_cross
        self.is_sparse = is_sparse
        self.struct = struct
        self.layer_num = cross_num
        self.num_experts = num_experts
        self.in_features = self.dense_feature_dim + self.sparse_num_field * self.sparse_feature_dim
        self.low_rank = low_rank
        self.net = net_struct
        self.init_value_ = 0.1
        self.soft = nn.Softmax()
    
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=True,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.KaimingNormal(
                 )))

        self.w = nn.Linear(self.in_features, self.in_features)

        self.V_list=self.U_list = paddle.create_parameter(
            shape=[self.layer_num,
            self.num_experts,
            self.in_features,
            self.low_rank
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.KaimingNormal(
            ))
        self.C_list = paddle.create_parameter(
            shape=[self.layer_num,
            self.num_experts,
            self.low_rank,
            self.low_rank
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.KaimingNormal(
            ))
        self.gating = nn.LayerList([nn.Linear(self.in_features, 1, bias_attr=False) for i in range(self.num_experts)])

        self.layer_b = paddle.create_parameter(
            shape=[
                self.dense_feature_dim + self.sparse_num_field *
                self.sparse_feature_dim
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.KaimingNormal(
            ))
   
        if self.struct == "stack":
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1],
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.KaimingNormal(
                    )))
        else:
            self.fc = paddle.nn.Linear(
                in_features=self.layer_sizes[-1] + self.sparse_num_field *
                self.sparse_feature_dim + self.dense_feature_dim,
                out_features=1,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.KaimingNormal(
                    )))

        # DNN
        self.num_field = self.dense_feature_dim + self.sparse_num_field * self.sparse_feature_dim
        sizes = [self.num_field] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        for i in range(len(self.layer_sizes)):  # + 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.KaimingNormal(
                )))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)           

    def _create_embedding_input(self, sparse_inputs, dense_inputs):

        sparse_inputs_concat = paddle.concat(
            sparse_inputs, axis=1)  

        sparse_embeddings = self.embedding(
            sparse_inputs_concat)  

        sparse_embeddings_re = paddle.reshape(
            sparse_embeddings,
            shape=[-1, self.sparse_num_field *
                   self.sparse_feature_dim])  
   
        feat_embeddings = paddle.concat([sparse_embeddings_re, dense_inputs],
                                        1)

        return feat_embeddings

    def _l2_loss(self, w):
        return paddle.sum(paddle.square(w))

    def _cross_layer(self, input_0, input_x):
        input_layer_0 = self.w(input_x)

        input_w =  paddle.multiply(input_0, input_layer_0)
        input_layer = paddle.add(input_w, input_x)

        return input_layer, input_w


    def _cross_net_v2(self, input, num_corss_layers):
        x = x0 = input

        l2_reg_cross_list = []
        for i in range(num_corss_layers):
            x, w = self._cross_layer(x0, x)
            l2_reg_cross_list.append(self._l2_loss(w))
        l2_reg_cross_loss = paddle.sum(
            paddle.concat(
                l2_reg_cross_list, axis=-1))
       
        return x, l2_reg_cross_loss

    def _cross_net_mix(self,input):
        x_l = x_0 = input 
        l2_reg_cross_list = []
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):

                gating_score_of_experts.append(self.gating[expert_id](x_l))
      
                v_x = paddle.matmul(self.V_list[i][expert_id].t(),x_l.t())
          
                v_x = paddle.tanh(v_x)
                v_x = paddle.matmul(self.C_list[i][expert_id], v_x)
                v_x = paddle.tanh(v_x)

                uv_x = paddle.matmul(self.U_list[i][expert_id], v_x)  

                dot_ = paddle.add(uv_x.t(), self.layer_b)    #[8, 247]   [247]
                dot_ = x_0 * dot_  

                output_of_experts.append(dot_)

            output_of_experts = paddle.stack(output_of_experts, 2)  
            gating_score_of_experts = paddle.stack(gating_score_of_experts, 1) 
            
            gating_score_of_experts = self.soft(gating_score_of_experts)
            moe_out = paddle.matmul(output_of_experts, gating_score_of_experts)

            moe_out = paddle.squeeze(moe_out)
            x_l = paddle.add(moe_out, x_l) 
            l2_reg_cross_list.append(self._l2_loss(moe_out))
        l2_reg_cross_loss = paddle.sum(
            paddle.concat(l2_reg_cross_list, axis=-1))
        return x_l, l2_reg_cross_loss

    def forward(self, sparse_inputs, dense_inputs):

        feat_embeddings = self._create_embedding_input(
            sparse_inputs, dense_inputs)  

        if self.net == "mix":
            cross_out, l2_reg_cross_loss = self._cross_net_mix(feat_embeddings)
        else:
            cross_out, l2_reg_cross_loss = self._cross_net_v2(feat_embeddings, self.cross_num)
        
        if self.struct == "stack":
            dnn_feat = cross_out
            for n_layer in self._mlp_layers:
                dnn_feat = n_layer(dnn_feat)

            last_out = dnn_feat
        else:
            dnn_feat = feat_embeddings
            for n_layer in self._mlp_layers:
                dnn_feat = n_layer(dnn_feat)

            last_out = paddle.concat([dnn_feat, cross_out], axis=-1)
     
        logit = self.fc(last_out)
        predict = F.sigmoid(logit)

        return predict, l2_reg_cross_loss