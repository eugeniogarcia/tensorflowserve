# Introduction

To load our trained model into TensorFlow Serving __we first need to save it in SavedModel format__. This __will create a protobuf file__ in a well-defined directory hierarchy, and will include a version number. __TensorFlow Serving allows us to select which version of a model__, or __"servable"__ we want to use __when we make inference requests__. Each version will be exported to a different sub-directory under the given path.

```py
tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
```

Once we have stored the model we can inspect it:

```sh
saved_model_cli show --dir C:/Users/Eugenio/AppData/Local/Temp/1 --all
```

```yml
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['Conv1_input'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 28, 28, 1)
        name: serving_default_Conv1_input:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['Softmax'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
WARNING:tensorflow:From C:/Users/Eugenio/AppData/Roaming/Python/Python37/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1785: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          Conv1_input: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='Conv1_input')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #3
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #4
      Callable with:
        Argument #1
          Conv1_input: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='Conv1_input')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          Conv1_input: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='Conv1_input')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          Conv1_input: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='Conv1_input')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #3
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #4
      Callable with:
        Argument #1
          Conv1_input: TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='Conv1_input')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
```

Note that it says that __Method name is: tensorflow/serving/predict__.

## Train and store the model

In `train_and_serve.py` we have an example where we do train a model, and then we store it as described above in order to be served later. In order to serve it with Tensorflow serve, we  start the container - we use docker for this example. We specify the path to the model we have store, and the name of the model in `MODEL_NAME`.

## Tensorflow server

```sh
docker run -t --rm -p 8501:8501 -v "C:/Users/Eugenio/AppData/Local/Temp/:/models/moda" -e MODEL_NAME=moda tensorflow/serving
```

## Consume the model

To consume the model the url would be: `http://localhost:8501/v1/models/moda/versions/1:predict`. Note that we are setting in the url the name of the model, `moda`, the version, `/versions/1` and the function to call `:predict`.

In `predict.py` we have an example of consuming the model.
