# encoding: utf-8
#used for CUB_200_2011   double output    adding an additional conv layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input,VGG16
from keras.layers import GlobalAveragePooling2D,Dense,Conv2D,Dropout
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad,Adam
import numpy as np
from keras.models import load_model  


# 数据准备
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)



train_generator = train_datagen.flow_from_directory(directory='./CUB_200_2011/CUB_200_2011/train',
                                  target_size=(224,224),#VGGNet16规定大小
                                  batch_size=32)            
val_generator = val_datagen.flow_from_directory(directory='./CUB_200_2011/CUB_200_2011/validation',
                                target_size=(224,224),
                                batch_size=32)   



# 构建基础模型
base_model = VGG16(weights='imagenet',include_top=False)

# 增加新的输出层
x = base_model.layers[-2].output

x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block6')(x)

x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数

# x = Dropout(0.2)(x)
predictions = Dense(200,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
# plot_model(model,'tlmodel.png')
model.summary()


def setup_to_transfer_learning(model):

    GAP_LAYER_ = 17 # max_pooling_2d_2
    for layer in model.layers[:GAP_LAYER_+1]:
        layer.trainable = False
    for layer in model.layers[GAP_LAYER_+1:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model):
    GAP_LAYER = 13 # max_pooling_2d_2
    for layer in model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])



setup_to_transfer_learning(model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=200,#800                     
                    epochs=10,#2
                    validation_data=val_generator,
                    validation_steps=6,#12
                    class_weight='auto'
                    )

# setup_to_fine_tune(model)
# history_ft = model.fit_generator(generator=train_generator,
#                                  steps_per_epoch=200,                
#                                  epochs=6,
#                                  validation_data=val_generator,
#                                  validation_steps=6,
#                                  class_weight='auto')



#保存中间模型以供继续训练
model.save('./mid_model.h5')

#提取并存储全连接层参数
w=model.layers[-1].get_weights()
w=w[0]
w=np.array(w)
np.save("./fcn_w.npy",w)

#改变model的结构，将feature map作为输出
model_save = Model(inputs=model.input,outputs=[model.layers[-3].output,model.output])
#保存模型
model_save.save('./cam_model.h5')
