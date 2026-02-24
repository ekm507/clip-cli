
download the models from here:

https://huggingface.co/ajaleksa/clip-onnx-models

put them into models/vision.onnx and models/text.onnx

then build this repo


this CLI has three commands:
`add` which adds an image to the database
`search` which searches a text from the images and retrieves `n` relevant images.
also an additional command `embed` which given any text or image, returns the embedding vector of that without searching or adding. 
