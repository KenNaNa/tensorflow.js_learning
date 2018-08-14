# TensorFlow.js中的核心概念
```
TensorFlow.js是一个用于机器智能的开源WebGL加速JavaScript库。
它为您的指尖带来了高性能的机器学习构建块，允许您在浏览器中训练神经网络或在推理模式下运行预先训练的模型。
有关安装/配置TensorFlow.js的指南，请参阅“ 入门”。

TensorFlow.js提供用于机器学习的低级构建块以及用于构建神经网络的高级Keras启发式API。
我们来看看该库的一些核心组件。
```
# 张量
```
TensorFlow.js中的中心数据单位是张量：一组数值，形状为一个或多个维度的数组。
甲Tensor实例有一个shape定义该阵列形状属性（即，有多少个值是在所述阵列的每一维）。

主要Tensor构造函数是tf.tensor函数：
```
```
// 2x3 Tensor
const shape = [2, 3]; // 2 rows, 3 columns
const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape);
a.print(); // print Tensor values
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]

// The shape can also be inferred:
const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
b.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]
```
然而，构建低秩张量，我们建议您使用以下功能来提高代码的可读性：
tf.scalar，tf.tensor1d，tf.tensor2d，tf.tensor3d和tf.tensor4d。

以下示例使用以下内容创建与上面相同的张量tf.tensor2d：
```
const c = tf.tensor2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
c.print();
// Output: [[1 , 2 , 3 ],
//          [10, 20, 30]]
TensorFlow.js还提供了方便的功能，用于创建张量，所有值设置为0（tf.zeros）或所有值设置为1（tf.ones）：

// 3x5 Tensor with all values set to 0
const zeros = tf.zeros([3, 5]);
// Output: [[0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0],
//          [0, 0, 0, 0, 0]]
```
在TensorFlow.js中，张量是不可变的; 一旦创建，您就无法更改其值。而是对它们执行生成新张量的操作。

# 变量
Variables用值张量初始化。Tensor然而，与s 不同，它们的值是可变的。
您可以使用以下assign方法为现有变量指定新的张量：
```
const initialValues = tf.zeros([5]);
const biases = tf.variable(initialValues); // initialize biases
biases.print(); // output: [0, 0, 0, 0, 0]

const updatedValues = tf.tensor1d([0, 1, 0, 1, 0]);
biases.assign(updatedValues); // update values of biases
biases.print(); // output: [0, 1, 0, 1, 0]
```
变量主要用于在模型训练期间存储然后更新值。

# 运营（行动）
虽然张量允许您存储数据，但操作（操作）允许您操作该数据。
TensorFlow.js提供了多种适用于线性代数和机器学习的运算，可以在张量上执行。
因为张量是不可变的，所以这些操作不会改变它们的值; 相反，ops返回新的张量。

可用的操作包括一元操作，例如square：
```
const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// Output: [[1, 4 ],
//          [9, 16]]
而如二进制OPS add，sub以及mul：

const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]]);
const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]]);

const e_plus_f = e.add(f);
e_plus_f.print();
// Output: [[6 , 8 ],
//          [10, 12]]
TensorFlow.js有一个可链接的API; 你可以在ops的结果上调用ops：

const sq_sum = e.add(f).square();
sq_sum.print();
// Output: [[36 , 64 ],
//          [100, 144]]

// All operations are also exposed as functions in the main namespace,
// so you could also do the following:
const sq_sum = tf.square(tf.add(e, f));

```
# 模型和图层
从概念上讲，模型是一种函数，给定一些输入将产生一些所需的输出。

在TensorFlow.js中，有两种方法可以创建模型。您可以直接使用ops来表示模型所做的工作。例如：
```
// Define function
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  // More on tf.tidy in the next section
  return tf.tidy(() => {
    const x = tf.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

// Define constants: y = 2x^2 + 4x + 8
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

// Predict output for input of 2
const result = predict(2);
result.print() // Output: 24
您还可以使用高级API tf.model来构建层模型，这是深度学习中的流行抽象。以下代码构造了一个tf.sequential模型：

const model = tf.sequential();
model.add(
  tf.layers.simpleRNN({
    units: 20,
    recurrentInitializer: 'GlorotNormal',
    inputShape: [80, 4]
  })
);

const optimizer = tf.train.sgd(LEARNING_RATE);
model.compile({optimizer, loss: 'categoricalCrossentropy'});
model.fit({x: data, y: labels});
```
TensorFlow.js中有许多不同类型的层。举几个例子包括tf.layers.simpleRNN，tf.layers.gru，和tf.layers.lstm。

# 内存管理：处理和tf.tidy
由于TensorFlow.js使用GPU来加速数学运算，因此在使用张量和变量时需要管理GPU内存。

TensorFlow.js提供了两个函数来帮助解决这个问题：dispose和tf.tidy。

# 部署
您可以调用dispose张量或变量来清除它并释放其GPU内存：
```
const x = tf.tensor2d([[0.0, 2.0], [4.0, 6.0]]);
const x_squared = x.square();

x.dispose();
x_squared.dispose();
tf.tidy
dispose在进行大量张量操作时使用可能很麻烦。TensorFlow.js提供了另一个函数，
tf.tidy它在JavaScript中扮演与常规作用域类似的角色，但对于GPU支持的张量。

tf.tidy执行一个函数并清除所有创建的中间张量，释放它们的GPU内存。它不会清除内部函数的返回值。

// tf.tidy takes a function to tidy up after
const average = tf.tidy(() => {
  // tf.tidy will clean up all the GPU memory used by tensors inside
  // this function, other than the tensor that is returned.
  //
  // Even in a short sequence of operations like the one below, a number
  // of intermediate tensors get created. So it is a good practice to
  // put your math ops in a tidy!
  const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
  const z = tf.ones([4]);

  return y.sub(z).square().mean();
});
average.print() // Output: 3.5

```
使用tf.tidy将有助于防止应用程序中的内存泄漏。它还可以用于更加谨慎地控制何时回收内存。

# 两个重要的注释
```
传递给的函数tf.tidy应该是同步的，也不会返回Promise。我们建议保留更新UI或在远程请求之外的代码tf.tidy。

tf.tidy 不会清理变量。变量通常持续到机器学习模型的整个生命周期，
因此TensorFlow.js即使它们是在一个中创建的，也不会清理它们tidy。
```
但是，您可以dispose手动调用它们。
