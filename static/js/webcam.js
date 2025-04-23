const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const annotated = document.getElementById('annotated');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// Send frame every second
setInterval(() => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const dataURL = canvas.toDataURL('image/jpeg');

  fetch('/predict-frame', {
    method: 'POST',
    body: new URLSearchParams({ 'frame': dataURL }),
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  })
  .then(res => res.json())
  .then(data => {
    annotated.src = "data:image/jpeg;base64," + data.annotated;
  });
}, 1000);
