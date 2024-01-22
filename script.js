
const viedo = document.getElementById('video');
const satrtbutton = document.getElementById('startButton')

let webcamStarted = false;


const startWebcam = (e) => {
  if (!webcamStarted) {
    // Stop any existing video tracks
    if (viedo.srcObject) {
      viedo.srcObject.getTracks().forEach(track => track.stop());
    }

    navigator.mediaDevices.getUserMedia({"video" : true , "audio" : false})
    .then((stream) => {
      viedo.srcObject = stream;
      webcamStarted = true;
      document.getElementById('myText').style.display = 'none';
    }).catch((err) => {
      console.error(err);
    });

  }
};



function getLabeledFaceDescriptions(){
  const labels = ['Messi','MinnMinn',"YinMinnmyat","MayZinThet","Joy"];
  return Promise.all(
    labels.map(async(label)=>{
      let descriptions = []; // Declare descriptions here
      for(let i=1; i<=2; i++){
        const img = await faceapi.fetchImage(`./Labels/${label}/${i}.jpg`);
        const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  )
}


async function faceRecognition(){
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  viedo.addEventListener('play',()=>{
    const canvas = faceapi.createCanvasFromMedia(viedo);
    document.body.append(canvas);

    const displaySize = { width: viedo.width, height: viedo.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      const detections = await faceapi
        .detectAllFaces(viedo)
        .withFaceLandmarks()
        .withFaceDescriptors();
  
      const resizedDetections = faceapi.resizeResults(detections, displaySize);
  
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  
      const results = resizedDetections.map((d) => {
        return faceMatcher.findBestMatch(d.descriptor);
      });
      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result,
        });
        drawBox.draw(canvas);
      });
    }, 100);
  })
};

Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
]).then(satrtbutton.addEventListener('click',startWebcam)).then(faceRecognition);







