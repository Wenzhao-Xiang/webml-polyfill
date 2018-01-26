describe('Mul Test', function() {
  const assert = chai.assert;
  const TENSOR_DIMENSIONS = [2, 2, 2, 2];
  const nn = navigator.ml.getNeuralNetworkContext();
  const value0 = 0.4;
  const value1 = 0.5;
    
  it('check result', async function() {
    let operandIndex = 0;
    let model = await nn.createModel();
    const float32TensorType = {type: nn.TENSOR_FLOAT32, dimensions: TENSOR_DIMENSIONS};
    const tensorLength = product(float32TensorType.dimensions);

    let fusedActivationFuncNone = operandIndex++;
    await model.addOperand({type: nn.INT32});
    await model.setOperandValue(fusedActivationFuncNone, new Int32Array([nn.FUSED_NONE]));

    let input0 = operandIndex++;
    await model.addOperand(float32TensorType);
    let input0Data = new Float32Array(tensorLength);
    input0Data.fill(value0);

    model.setOperandValue(input0, input0Data);

    let input1 = operandIndex++;
    await model.addOperand(float32TensorType);
    let output = operandIndex++;
    await model.addOperand(float32TensorType);

    await model.addOperation(nn.MUL, [input0, input1, fusedActivationFuncNone], [output]);
    await model.identifyInputsAndOutputs([input1], [output]);
    await model.finish();

    let compilation = await model.createCompilation();

    await compilation.setPreference(nn.PREFER_FAST_SINGLE_ANSWER);
    
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input1Data = new Float32Array(tensorLength);
    input1Data.fill(value1);

    await execution.setInput(0, input1Data);

    let outputData = new Float32Array(tensorLength);
    await execution.setOutput(0, outputData);

    await execution.startCompute();

    for (let i = 0; i < tensorLength; ++i) {
      assert.isTrue(almostEqual(outputData[i], input0Data[i] * input1Data[i]));
    }
  });
});