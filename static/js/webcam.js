const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const annotated = document.getElementById('annotated');
const debugCanvas = document.getElementById('debugCanvas');
const debugCtx = debugCanvas.getContext('2d');

// Connect WebSocket
const socket = io();

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
});

// Send frame every second
setInterval(() => {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    debugCanvas.width = canvas.width;
    debugCanvas.height = canvas.height;
    debugCtx.drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL('image/jpeg');
    socket.emit('frame', dataURL);
  }
}, 1000);

// Receive prediction from server
socket.on('prediction', data => {
  annotated.src = "data:image/jpeg;base64," + data.annotated;
  document.getElementById('liveResult').innerText = data.results.join(', ');
});

socket.on('error', data => {
  console.error('Prediction error:', data.error);
});
