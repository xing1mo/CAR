
# ================================================================
# MIT License
# Copyright (c) 2022 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
from iseg.layers.normalizations import normalization

from iseg.utils.attention_utils import *
from iseg.layers.model_builder import resize_image, get_training_value
from iseg.vis.vismanager import get_visualization_manager

from car_core.utils import (
    get_flatten_one_hot_label,
    get_class_sum_features_and_counts,
    get_inter_class_relative_loss,
    get_intra_class_absolute_loss,
    get_pixel_inter_class_relative_loss,
)


class ClassAwareRegularization(tf.keras.Model):
    def __init__(
        self,
        train_mode=False,
        use_inter_class_loss=True,
        use_intra_class_loss=True,
        intra_class_loss_remove_max=False,
        use_inter_c2c_loss=True,
        use_inter_c2p_loss=False,
        intra_class_loss_rate=1,
        inter_class_loss_rate=1,
        num_class=21,
        ignore_label=0,
        pooling_rates=[1],
        use_batch_class_center=True,
        use_last_class_center=False,
        last_class_center_decay=0.9,
        inter_c2c_loss_threshold=0.5,
        inter_c2p_loss_threshold=0.25,
        filters=512,
        apply_convs=False,
        name=None,
    ):

        super().__init__(name=name)

        self.vis_manager = get_visualization_manager()

        self.train_mode = train_mode
        self.use_inter_class_loss = use_inter_class_loss
        self.use_intra_class_loss = use_intra_class_loss
        self.intra_class_loss_rate = intra_class_loss_rate
        self.inter_class_loss_rate = inter_class_loss_rate
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.inter_c2c_loss_threshold = inter_c2c_loss_threshold
        self.inter_c2p_loss_threshold = inter_c2p_loss_threshold

        self.intra_class_loss_remove_max = intra_class_loss_remove_max


        self.use_inter_c2c_loss = use_inter_c2c_loss
        self.use_inter_c2p_loss = use_inter_c2p_loss

        self.filters = filters
        self.apply_convs = apply_convs

        if isinstance(pooling_rates, tuple):
            pooling_rates = list(pooling_rates)

        if not isinstance(pooling_rates, list):
            pooling_rates = [pooling_rates]

        self.pooling_rates = pooling_rates
        self.use_batch_class_center = use_batch_class_center
        self.use_last_class_center = use_last_class_center
        self.last_class_center_decay = last_class_center_decay

        print(f"------CAR settings------")
        print(f"------train_mode = {train_mode}")
        print(f"------use_intra_class_loss = {use_intra_class_loss}")
        print(f"------use_inter_class_loss = {use_inter_class_loss}")
        print(f"------intra_class_loss_rate = {intra_class_loss_rate}")
        print(f"------inter_class_loss_rate = {inter_class_loss_rate}")

        print(f"------use_batch_class_center = {use_batch_class_center}")
        print(f"------use_last_class_center = {use_last_class_center}")
        print(f"------last_class_center_decay = {last_class_center_decay}")

        print(f"------pooling_rates = {pooling_rates}")
        print(f"------inter_c2c_loss_threshold = {inter_c2c_loss_threshold}")
        print(f"------inter_c2p_loss_threshold = {inter_c2p_loss_threshold}")

        print(f"------intra_class_loss_remove_max = {intra_class_loss_remove_max}")

        print(f"------use_inter_c2c_loss = {use_inter_c2c_loss}")
        print(f"------use_inter_c2p_loss = {use_inter_c2p_loss}")

        print(f"------filters = {filters}")
        print(f"------apply_convs = {apply_convs}")

        print(f"------num_class = {num_class}")
        print(f"------ignore_label = {ignore_label}")


# features：形状为 [N, H, W, C] 的特征张量，其中 N 表示批次大小，H 表示图像的高度，W 表示图像的宽度，C 表示特征通道数。
# label：形状为 [N, H, W, 1] 的标签张量，其中每个像素对应的值表示其所属的类别。
# extra_prefix：一个字符串，表示在损失函数名字前额外添加的前缀。
# training：一个布尔值，表示模型当前是否处于训练模式。
    def add_car_losses(self, features, label=None, extra_prefix=None, training=None):

        # features : [N, H, W, C]

        training = get_training_value(training)

# loss_name_prefix 表示损失函数的名字前缀，extra_prefix 是一个可选参数，用于在损失函数名字前额外添加前缀。
        loss_name_prefix = f"{self.name}"

        if extra_prefix is not None:
            loss_name_prefix = f"{loss_name_prefix}_{extra_prefix}"

#inputs_shape 变量表示 features 的形状，height 和 width 变量表示图像的高度和宽度
        inputs_shape = tf.shape(features)
        height = inputs_shape[-3]
        width = inputs_shape[-2]

#对标签进行了处理，将其缩放到与特征张量相同的大小，并进行了一些检查，确保特征张量中不包含 nan 或 inf。
        label = resize_image(label, (height, width), method="nearest")

        tf.debugging.check_numerics(features, "features contains nan or inf")

#将特征张量展平成二维数组
        flatten_features = flatten_hw(features)

#创建了一个掩码，表示哪些像素不应该被忽略
        not_ignore_spatial_mask = tf.cast(label, tf.int32) != self.ignore_label  # [N, H, W, 1]
        not_ignore_spatial_mask = flatten_hw(not_ignore_spatial_mask)

#one_hot_label 是一个二维张量，其中每一行表示一个像素的类别独热编码
        one_hot_label = get_flatten_one_hot_label(
            label, num_class=self.num_class, ignore_label=self.ignore_label
        )  # [N, HW, class]

        ####################################################################################

#class_sum_features为真值类中心(每个类别特征值的总和），class_sum_features为非零元素数量
        class_sum_features, class_sum_non_zero_map = get_class_sum_features_and_counts(
            flatten_features, one_hot_label
        )  # [N, class, C]

#为了减轻噪声图像的负面影响，我们使用批处理(a batch)的所有训练图像计算类中心
        if self.use_batch_class_center:

            #获取replica_context，如果replica_context存在，则表示处于分布式环境下，需要对计算结果进行全局合并
            replica_context = tf.distribute.get_replica_context()

            #对不同batch进行求和
            class_sum_features_in_cross_batch = tf.reduce_sum(
                class_sum_features, axis=0, keepdims=True, name="class_sum_features_in_cross_batch"
            )
            class_sum_non_zero_map_in_cross_batch = tf.reduce_sum(
                class_sum_non_zero_map, axis=0, keepdims=True, name="class_sum_non_zero_map_in_cross_batch"
            )
            #如果处于分布式环境下，进行全局合并
            if replica_context:
                class_sum_features_in_cross_batch = replica_context.all_reduce(
                    tf.distribute.ReduceOp.SUM, class_sum_features_in_cross_batch
                )
                class_sum_non_zero_map_in_cross_batch = replica_context.all_reduce(
                    tf.distribute.ReduceOp.SUM, class_sum_non_zero_map_in_cross_batch
                )

            #如公式1，除上非零数量
            class_avg_features_in_cross_batch = tf.math.divide_no_nan(
                class_sum_features_in_cross_batch, class_sum_non_zero_map_in_cross_batch
            )  # [1, class, C]

            if self.use_last_class_center:#是否需要考虑上次迭代得到的类中心
                #通过比较class_sum_non_zero_map_in_cross_batch和零得到一个布尔掩码,用于指示哪些类别在此batch中出现过
                batch_class_ignore_mask = tf.cast(class_sum_non_zero_map_in_cross_batch != 0, tf.int32)
                
                #将class_avg_features_in_cross_batch和self.last_class_center相减，得到每个类别特征向量的差异
                class_center_diff = class_avg_features_in_cross_batch - tf.cast(self.last_class_center, class_avg_features_in_cross_batch.dtype)
                #提取class_center_diff中所有存在的类别乘上系数last_class_center_decay，得到需要保留的diff
                class_center_diff *= (1 - self.last_class_center_decay) * tf.cast(batch_class_ignore_mask, class_center_diff.dtype)

                #计算出本次迭代的last_class_center
                self.last_class_center.assign_add(class_center_diff)

                class_avg_features_in_cross_batch = tf.cast(self.last_class_center, tf.float32)

            class_avg_features = class_avg_features_in_cross_batch

        else:
            class_avg_features = tf.math.divide_no_nan(
                class_sum_features, class_sum_non_zero_map
            )  # [N, class, C]

        ####################################################################################

        if self.use_inter_class_loss and training:#计算类间loss

            inter_class_relative_loss = 0

            if self.use_inter_c2c_loss:
                inter_class_relative_loss += get_inter_class_relative_loss(#最大化类中心之间的距离
                    class_avg_features, inter_c2c_loss_threshold=self.inter_c2c_loss_threshold,
                )

            if self.use_inter_c2p_loss:
                inter_class_relative_loss += get_pixel_inter_class_relative_loss(#最大化类中心与不属于该类的任何像素之间的距离
                    flatten_features, class_avg_features, one_hot_label, inter_c2p_loss_threshold=self.inter_c2p_loss_threshold,
                )

            self.add_loss(inter_class_relative_loss * self.inter_class_loss_rate)
            self.add_metric(inter_class_relative_loss, name=f"{loss_name_prefix}_orl")

        if self.use_intra_class_loss:#计算类内loss

            same_avg_value = tf.matmul(one_hot_label, class_avg_features)

            tf.debugging.check_numerics(same_avg_value, "same_avg_value contains nan or inf")

            self_absolute_loss = get_intra_class_absolute_loss(
                flatten_features,
                same_avg_value,
                remove_max_value=self.intra_class_loss_remove_max,
                not_ignore_spatial_mask=not_ignore_spatial_mask,
            )

            if training:
                self.add_loss(self_absolute_loss * self.intra_class_loss_rate)
                self.add_metric(self_absolute_loss, name=f"{loss_name_prefix}_sal")

            print("Using self-loss")

    def build(self, input_shape):

        # Note that, this is not the best design for specified architecture, but a trade-off for generalizability

        channels = input_shape[0][-1]
        channels = self.filters if channels > self.filters else channels

        print(f"car channels = {channels}")

        self.linear_conv = tf.keras.layers.Conv2D(channels, (1, 1), use_bias=True, name="linear_conv",)

        if self.apply_convs:
            self.end_conv = tf.keras.layers.Conv2D(channels, (1, 1), use_bias=False, name="end_conv",)
            self.end_norm = normalization(name="end_norm")

        if self.use_last_class_center:
            self.last_class_center = self.add_weight(
                name="last_class_center",
                shape=[1, self.num_class, channels],
                dtype=tf.float32,
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=False,
            )
            

    def call(self, inputs, training=None):

        inputs, label = inputs

        x = inputs

        # This linear conv (w/o norm&activation) can be merged 
        # to the next one (end_conv) during inference
        # Simple (x * w0 + b) * w1 dot product
        # We keep it for better understanding
        x = self.linear_conv(x) #一个conv

        y = tf.identity(x)#把任意输入强制转换成 Tensor

        if self.train_mode and get_training_value(training):#训练模式直接监督

            x = tf.cast(x, tf.float32)

            tf.debugging.check_numerics(x, "inputs contains nan or inf")

            num_pooling_rates = len(self.pooling_rates)

            for i in range(num_pooling_rates):

                pooling_rate = self.pooling_rates[i]

                sub_x = tf.identity(x, name=f"x_in_rate_{pooling_rate}")

                if pooling_rate > 1:
                    stride_size = (1, pooling_rate, pooling_rate, 1)
                    sub_x = tf.nn.avg_pool2d(sub_x, stride_size, stride_size, padding="SAME")#平均池化

                self.add_car_losses(sub_x, label=label, extra_prefix=str(pooling_rate), training=training)

        if self.apply_convs:
            y = self.end_conv(y)
            y = self.end_norm(y, training=training)
            y = tf.nn.relu(y)

        return y
