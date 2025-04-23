const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const annotated = document.getElementById('annotated');
const debugCanvas = document.getElementById('debugCanvas');
const debugCtx = debugCanvas.getContext('2d');

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// Send frame every second
setInterval(() => {
  // Set canvas size to match video
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // Draw frame on hidden canvas for prediction
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  // DEBUG: draw timestamp
  ctx.fillStyle = "red";
  ctx.font = "16px Arial";
  ctx.fillText(new Date().toLocaleTimeString(), 10, 20);

  // DEBUG: also draw to visible debug canvas
  debugCanvas.width = canvas.width;
  debugCanvas.height = canvas.height;
  debugCtx.drawImage(video, 0, 0);

  const dataURL = canvas.toDataURL('image/jpeg');

  // Send frame to server
  fetch('/predict-frame', {
    method: 'POST',
    body: new URLSearchParams({ 'frame': dataURL }),
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  })
  .then(res => res.json())
  .then(data => {
    annotated.src = "data:image/jpeg;base64," + data.annotated;
    document.getElementById('liveResult').innerText = data.results.join(', ');
  });
}, 1000);
