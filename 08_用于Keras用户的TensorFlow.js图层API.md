```
TensorFlow.js的Layers API以Keras为模型。正如您可能已经从教程和示例中注意到的 那样，考虑到JavaScript和Python之间的差异，我们努力使 Layers API与Keras类似。这使得具有使用Python开发Keras模型的经验的用户可以更轻松地迁移到JavaScript中的TensorFlow.js图层。例如，以下Keras代码转换为JavaScript：

# Python:
import keras
import numpy as np

# Build and compile model.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generate some synthetic data for training.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# Train model with fit().
model.fit(xs, ys, epochs=1000)

# Run inference with predict().
print(model.predict(np.array([[5]])))
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

// Build and compile model.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// Train model with fit().
await model.fit(xs, ys, {epochs: 1000});

// Run inference with predict().
model.predict(tf.tensor2d([[5]], [1, 1])).print();
但是，我们希望在本文档中提及并解释一些差异。
一旦了解了这些差异及其背后的基本原理，您的Python到JavaScript迁移（或反向迁移）应该是一种相对平稳的体验。

构造函数将JavaScript对象作为配置
比较上面示例中的以下Python和JavaScript行：它们都创建了一个Dense图层。

# Python:
keras.layers.Dense(units=1, inputShape=[1])
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
JavaScript函数在Python函数中没有等效的关键字参数。
我们希望避免在JavaScript中将构造函数选项实现为位置参数，
这对于记忆和使用具有大量关键字参数的构造函数（例如，LSTM）来说尤其麻烦 。
这就是我们使用JavaScript配置对象的原因。这些对象提供与Python关键字参数相同的位置不变性和灵活性。

例如，Model类的一些方法 Model.compile()也将JavaScript配置对象作为输入。
但是，请记住 Model.fit()，Model.evaluate()并且Model.predict()略有不同。
由于这些方法将强制性x（特征）和y（标签或目标）数据作为输入; x并且y是与后续配置对象分开的位置参数，该配置对象扮演关键字参数的角色。例如：

// JavaScript:
await model.fit(xs, ys, {epochs: 1000});
Model.fit（）是异步的
Model.fit()是用户在TensorFlow.js中执行模型训练的主要方法。这种方法通常可以长时间运行，
持续数秒或数分钟。因此，我们利用asyncJavaScript语言的特性，
以便在浏览器中运行时可以以不阻塞主UI线程的方式使用此函数。
这类似于JavaScript中其他可能长时间运行的函数，例如async fetch。
请注意，这async是Python中不存在的构造。虽然fit() Keras中的 方法返回一个History对象，
但fit()JavaScript中方法的对应部分返回一个 Promise of History，可以 等待 ed（如上例所示）或者使用then（）方法。

TensorFlow.js没有NumPy
Python Keras用户经常使用NumPy来执行基本的数值和数组操作，例如在上面的示例中生成2D张量。

# Python:
xs = np.array([[1], [2], [3], [4]])
在TensorFlow.js中，这种基本的数字操作是使用包本身完成的。例如：

// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
该tf.*命名空间还提供了许多其它功能阵列和线性代数运算如矩阵乘法。有关更多信息，请参阅 TensorFlow.js核心文档。

使用工厂方法，而不是构造函数
Python中的这一行（来自上面的例子）是一个构造函数调用：

# Python:
model = keras.Sequential()
如果严格转换为JavaScript，则等效构造函数调用将如下所示：

// JavaScript:
const model = new tf.Sequential();  // !!! DON'T DO THIS !!!
但是，我们决定不使用“新”构造函数，因为1）“new”关键字会使代码更加膨胀; 2）
“新”构造函数被视为JavaScript的“坏部分”：潜在的陷阱，如在JavaScript中争论 ：好的部分。
要在TensorFlow.js中创建模型和图层，可以调用具有lowerCamelCase名称的工厂方法，例如：

// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
选项字符串值为lowerCamelCase，而不是snake_case
在JavaScript中，与Python相比，使用驼峰案例更常见的是符号名称（例如，参见 Google JavaScript样式指南），
其中蛇案例很常见（例如，在Keras中）。因此，我们决定使用lowerCamelCase作为选项的字符串值，包括以下内容：

DataFormat，例如，channelsFirst而不是channels_first
初始化器，例如，glorotNormal而不是glorot_normal
损失和指标，例如，meanSquaredError而不是 mean_squared_error，distincticalCrossentropy而不是 categorical_crossentropy。
例如，如上例所示：

// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
关于模型序列化和反序列化，请放心。TensorFlow.js的内部机制确保正确处理JSON对象中的蛇案例，例如，从Python Keras加载预训练模型时。

使用apply（）运行Layer对象，而不是将它们作为函数调用
在Keras中，Layer对象具有__call__定义的方法。因此，用户可以通过将对象作为函数调用来调用图层的逻辑，例如，

# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
这个Python语法糖作为TensorFlow.js中的apply（）方法实现：

// JavaScript:
const myInput = tf.input{shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
Layer.apply（）支持对具体张量的命令式（急切）评估
目前，在Keras中，call方法只能对（Python）TensorFlow的Tensor 对象（假设TensorFlow后端）进行操作，
这些对象是符号的并且不包含实际的数值。这就是上一节中的示例所示。
但是，在TensorFlow.js中，层的apply（）方法可以在符号和命令模式下运行。
如果apply()使用SymbolicTensor（类似于tf.Tensor）调用，则返回值将为SymbolicTensor。这通常在模型构建期间发生。
但是如果apply()使用实际的具体Tensor值调用它，它将返回一个具体的Tensor。例如：

// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
此功能让人联想到（Python）TensorFlow的 Eager Execution。
除了打开组成动态神经网络的大门外，它还在模型开发过程中提供了更强的交互性和可调试性。

优化者正在训练中。，不是优化者。
在Keras中，Optimizer对象的构造函数位于 keras.optimizers.*命名空间下。在TensorFlow.js图层中，
优化程序的工厂方法位于tf.train.*命名空间下。例如：

# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
loadModel（）从URL加载，而不是从HDF5文件加载
在Keras中，模型通常保存 为HDF5（.h5）文件，以后可以使用该keras.models.load_model()方法加载 。
该方法采用.h5文件的路径。load_model()TensorFlow.js中的对应物是 tf.loadModel()。
由于HDF5不是浏览器友好的文件格式，因此tf.loadModel()采用TensorFlow.js特定的格式。
tf.loadModel()将model.json文件作为其输入参数。可以使用tensorflowjs pip包从Keras HDF5文件转换model.json。

// JavaScript:
const model = await tf.loadModel('https://foo.bar/model.json');
还要注意的是tf.loadModel()返回 Promise 的tf.Model。

通常，tf.Model在TensorFlow.js中保存和加载s 分别使用tf.Model.save和tf.loadModel方法完成 。
我们将这些API设计为类似于 Keras的save和load_model API 。但是浏览器环境与像Keras这样的主要深度学习框架运行的后端环境完全不同，
特别是在用于持久化和传输数据的路由数组中。因此，TensorFlow.js和Keras中的save / load API之间存在一些有趣的差异。有关更多详细信息，
请参阅我们关于保存和加载tf.Model的教程。
```
