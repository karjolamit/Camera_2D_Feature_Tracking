# Camera_2D_Feature_Tracking

## Data Buffer Optimization
Loading set of images using Data Buffer vector (Ring buffer). Adding new images to tail (right) and removing old images from head (left).
![Plot for 1D FFT Range measurement](https://github.com/karjolamit/Radar-Target-Generation-and-Detetction/blob/master/Plot%20for%201D%20FFT%20Range%20measurement.png)
```
DataFrame frame;
frame.cameraImg = imgGray;
dataBuffer.push_back(frame);
if (dataBuffer.size() > dataBufferSize) dataBuffer.pop_front();
assert(dataBuffer.size() <= dataBufferSize);
```
