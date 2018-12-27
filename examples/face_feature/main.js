const face_landmark_tflite = {
  modelName: 'Face Landmark(tflite)',
  modelFile: './model/face_landmark.tflite',
  inputSize: [128, 128, 3],
  outputSize: 136,
  preOptions: {
    // mean and std should also be in BGR order
    // norm: true,
  },
};
const face_tflite = {
  modelName: 'Face Detec.(TFlite)',
  inputSize: [416, 416, 3],
  outputSize: 1 * 13 * 13 * 30,
  modelFile: './model/yolov2_tiny-face.tflite',
  preOptions: {
    // mean and std should also be in BGR order
    norm: true,
  },
};

const preferMap = {
  'MPS': 'sustained',
  'BNNS': 'fast',
  'sustained': 'MPS',
  'fast': 'BNNS',
};

async function main(camera) {
  const availableModels = [
    face_tflite,
    face_landmark_tflite,
  ];

  const videoElement = document.getElementById('video');
  const imageElement = document.getElementById('image');
  const inputElement = document.getElementById('input');
  const buttonEelement = document.getElementById('button');
  const backend = document.getElementById('backend');
  const selectModel = document.getElementById('selectModel');
  const wasm = document.getElementById('wasm');
  const webgl = document.getElementById('webgl');
  const webml = document.getElementById('webml');
  const canvasElement = document.getElementById('canvas');
  const canvasElement1 = document.getElementById('canvas1');
  const canvasElement2 = document.getElementById('outputCanvas');
  const progressContainer = document.getElementById('progressContainer');
  const progressBar = document.getElementById('progressBar');
  const selectPrefer = document.getElementById('selectPrefer');

  let currentBackend = '';
  let currentModel = '';
  let currentPrefer = '';
  let streaming = false;

  let utils = new Utils(canvasElement);
  let utils1 = new Utils(canvasElement1);
  utils.updateProgress = updateProgress;    //register updateProgress function if progressBar element exist

  function checkPreferParam() {
    if (getOS() === 'Mac OS') {
      let preferValue = getPreferParam();
      if (preferValue === 'invalid') {
        console.log("Invalid prefer, prefer should be 'fast' or 'sustained', try to use WASM.");
        showPreferAlert();
      }
    }
  }

  checkPreferParam();

  function showAlert(backend) {
    let div = document.createElement('div');
    div.setAttribute('id', 'backendAlert');
    div.setAttribute('class', 'alert alert-warning alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Failed to setup ${backend} backend.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function showPreferAlert() {
    let div = document.createElement('div');
    div.setAttribute('id', 'preferAlert');
    div.setAttribute('class', 'alert alert-danger alert-dismissible fade show');
    div.setAttribute('role', 'alert');
    div.innerHTML = `<strong>Invalid prefer, prefer should be 'fast' or 'sustained'.</strong>`;
    div.innerHTML += `<button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>`;
    let container = document.getElementById('container');
    container.insertBefore(div, container.firstElementChild);
  }

  function removeAlertElement() {
    let backendAlertElem =  document.getElementById('backendAlert');
    if (backendAlertElem !== null) {
      backendAlertElem.remove();
    }
    let preferAlertElem =  document.getElementById('preferAlert');
    if (preferAlertElem !== null) {
      preferAlertElem.remove();
    }
  }

  function updateBackend() {
    if (getUrlParams('api_info') === 'true') {
      backend.innerHTML = currentBackend === 'WebML' ? currentBackend + '/' + getNativeAPI() : currentBackend;
    } else {
      backend.innerHTML = currentBackend;
    }
  }

  function changeBackend(newBackend, force) {
    if (!force && currentBackend === newBackend) {
      return;
    }
    streaming = false;
    if (newBackend !== "WebML") {
      selectPrefer.style.display = 'none';
    } else {
      selectPrefer.style.display = 'inline';
    }
    utils.deleteAll();
    backend.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(newBackend, currentPrefer).then(() => {
        currentBackend = newBackend;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${newBackend}, switch back to ${currentBackend}`);
        console.log(e);
        showAlert(newBackend);
        changeBackend(currentBackend, true);
        updatePrefer();
        updateModel();
        updateBackend();
      });
    }, 10);
  }

  function changeModel(newModel) {
    if (currentModel === newModel.modelName) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    utils.changeModelParam(newModel);
    progressContainer.style.display = "inline";
    currentPrefer = "sustained";
    selectModel.innerHTML = 'Setting...';
    currentModel = newModel.modelName;
    setTimeout(() => {
      utils.init(currentBackend, currentPrefer).then(() => {
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      });
    }, 10);
  }

  function updateModel() {
    selectModel.innerHTML = currentModel;
  }

  function changePrefer(newPrefer, force) {
    if (currentPrefer === newPrefer && !force) {
      return;
    }
    streaming = false;
    utils.deleteAll();
    selectPrefer.innerHTML = 'Setting...';
    setTimeout(() => {
      utils.init(currentBackend, newPrefer).then(() => {
        currentPrefer = newPrefer;
        updatePrefer();
        updateModel();
        updateBackend();
        if (!camera) {
          utils.predict(imageElement).then(ret => updateResult(ret));
        } else {
          streaming = true;
          startPredict();
        }
      }).catch((e) => {
        console.warn(`Failed to change backend ${preferMap[newPrefer]}, switch back to ${preferMap[currentPrefer]}`);
        console.error(e);
        showAlert(preferMap[newPrefer]);
        changePrefer(currentPrefer, true);
        updatePrefer();
        updateModel();
        updateBackend();
      });
    }, 10);
  }

  function updatePrefer() {
    selectPrefer.innerHTML = preferMap[currentPrefer];
  }

  function fileExists(url) {
    var exists;
    $.ajax({
      url:url,
      async:false,
      type:'HEAD',
      error:function() { exists = 0; },
      success:function() { exists = 1; }
    });
    if (exists === 1) {
      return true;
    } else {
      return false;
    }
  }

  function updateProgress(ev) {
    if (ev.lengthComputable) {
      let percentComplete = ev.loaded / ev.total * 100;
      percentComplete = percentComplete.toFixed(0);
      progressBar.style = `width: ${percentComplete}%`;
      progressBar.innerHTML = `${percentComplete}%`;
      if (ev.loaded === ev.total) {
        progressContainer.style.display = "none";
        progressBar.style = `width: 0%`;
        progressBar.innerHTML = `0%`;
      }
    }
  }
  //this.canvasContext.drawImage(imageSource, 94, 0,  face_1
   // 368,
   // 333, 0, 0, 128, 128);
  // this.canvasContext.drawImage(imageSource, 31, 0,  face_2
   // 412,
   // 347, 0, 0, 128, 128);
  // this.canvasContext.drawImage(imageSource, 5, 18,  face_6
  //  629,
  //  650, 0, 0, 128, 128);
  //this.canvasContext.drawImage(imageSource, 152, 4,  face_7
  //  70,
  //  70, 0, 0, 128, 128);
  function updateResult(result) {
    console.log(`Inference time: ${result.time} ms`);
    let inferenceTimeElement = document.getElementById('inferenceTime');
    inferenceTimeElement.innerHTML = `inference time: <em style="color:green;font-weight:bloder;">${result.time} </em>ms`;
    if (imageElement) {
      console.log(result.classes);
      // drawKeyPoints(imageElement, canvasElement2, result.classes);
      let out = decodeYOLOv2(result.classes, imageElement.width, imageElement.height);
      drawOutput(imageElement, canvasElement2, out, imageElement.width, imageElement.height);
    } else {
      let out = decodeYOLOv2(result.classes, videoElement.videoWidth, videoElement.videoHeight);
      drawOutput(videoElement, canvasElement2, out, videoElement.videoWidth, videoElement.videoHeight);
    }

  }
 
  if (nnNative) {
    webml.setAttribute('class', 'dropdown-item');
    webml.onclick = function (e) {
      removeAlertElement();
      checkPreferParam();
      changeBackend('WebML');
    };
  }

  if (nnPolyfill.supportWebGL) {
    webgl.setAttribute('class', 'dropdown-item');
    webgl.onclick = function(e) {
      removeAlertElement();
      changeBackend('WebGL');
    };
  }

  if (nnPolyfill.supportWasm) {
    wasm.setAttribute('class', 'dropdown-item');
    wasm.onclick = function(e) {
      removeAlertElement();
      changeBackend('WASM');
    };
  }

  if (currentBackend === '') {
    if (nnNative) {
      currentBackend = 'WebML';
    } else {
      currentBackend = 'WebGL';
    }
  }

  // register models
  for (let model of availableModels) {
    if (!fileExists(model.modelFile)) {
      continue;
    }
    let dropdownBtn = $('<button class="dropdown-item"/>')
      .text(model.modelName)
      .click(_ => changeModel(model));
    $('.available-models').append(dropdownBtn);
    if (!currentModel) {
      utils.changeModelParam(model);
      currentModel = model.modelName;
    }
  }
  utils1.changeModelParam(face_landmark_tflite);

  // register prefers
  if (getOS() === 'Mac OS' && currentBackend === 'WebML') {
    $('.prefer').css("display","inline");
    let MPS = $('<button class="dropdown-item"/>')
      .text('MPS')
      .click(_ => changePrefer(preferMap['MPS']));
    $('.preference').append(MPS);
    let BNNS = $('<button class="dropdown-item"/>')
      .text('BNNS')
      .click(_ => changePrefer(preferMap['BNNS']));
    $('.preference').append(BNNS);
    if (!currentPrefer) {
      currentPrefer = "sustained";
    }
  }

  // image or camera
  if (!camera) {
    inputElement.addEventListener('change', (e) => {
      let files = e.target.files;
      if (files.length > 0) {
        imageElement.src = URL.createObjectURL(files[0]);
      }
    }, false);

    imageElement.onload = async function() {
      let result = await utils.predict(imageElement); //.then(ret => {
        let out = decodeYOLOv2(result, imageElement.width, imageElement.height);
        let face_boxes = getFaceBoxes(out, imageElement.width, imageElement.height);
        let keyPoints = [];
        for (let i = 0; i< face_boxes.length; ++i) {
          let tmp = await utils1.predict(imageElement, face_boxes[i]);
          keyPoints.push(tmp);
        }
        drawFaceBoxes(imageElement, canvasElement2, face_boxes);
        drawKeyPoints(canvasElement2, keyPoints, face_boxes);
        // updateResult(ret);
      //});
    }
    await utils.init(currentBackend, currentPrefer);// .then(() => {
    await utils1.init(currentBackend, currentPrefer);
    updateBackend();
    updateModel();
    updatePrefer();
    let result = await utils.predict(imageElement); //.then(ret => {
        let out = decodeYOLOv2(result, imageElement.width, imageElement.height);
        let face_boxes = getFaceBoxes(out, imageElement.width, imageElement.height);
        let keyPoints = [];
        for (let i = 0; i< face_boxes.length; ++i) {
          let tmp = await utils1.predict(imageElement, face_boxes[i]);
          keyPoints.push(tmp);
        }
        drawFaceBoxes(imageElement, canvasElement2, face_boxes);
        drawKeyPoints(canvasElement2, keyPoints, face_boxes);
        // updateResult(ret);
      // });
      buttonEelement.setAttribute('class', 'btn btn-primary');
      inputElement.removeAttribute('disabled');
    //}).catch((e) => {
    //  console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
    //  console.error(e);
    //  showAlert(utils.model._backend);
    //  changeBackend('WASM');
    //});
  } else {
    let stats = new Stats();
    stats.dom.style.cssText = 'position:fixed;top:60px;left:10px;cursor:pointer;opacity:0.9;z-index:10000';
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild(stats.dom);

    navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "environment"}}).then(async function(stream) {
      video.srcObject = stream;
      await utils.init(currentBackend, currentPrefer);// .then(() => {
      await utils1.init(currentBackend, currentPrefer);
        updateBackend();
        updateModel();
        updatePrefer();
        streaming = true;
        startPredict();
      //}).catch((e) => {
      //  console.warn(`Failed to init ${utils.model._backend}, try to use WASM`);
       // console.error(e);
       // showAlert(utils.model._backend);
       // changeBackend('WASM');
    //  });
    }).catch((error) => {
      console.log('getUserMedia error: ' + error.name, error);
    });

    async function startPredict() {
      if (streaming) {
        stats.begin();
        //utils.predict(videoElement).then(ret => updateResult(ret)).then(() => {
          let result = await utils.predict(videoElement); //.then(ret => {
        let out = decodeYOLOv2(result, videoElement.videoWidth, videoElement.videoHeight);
        let face_boxes = getFaceBoxes(out, videoElement.videoWidth, videoElement.videoHeight);
        let keyPoints = [];
        for (let i = 0; i< face_boxes.length; ++i) {
          let tmp = await utils1.predict(videoElement, face_boxes[i]);
          keyPoints.push(tmp);
        }
        drawFaceBoxes(videoElement, canvasElement2, face_boxes);
        drawKeyPoints(canvasElement2, keyPoints, face_boxes);
          stats.end();
          setTimeout(startPredict, 0);
        //});
      }
    }
  }
}
