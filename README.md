# SELF-SUPERVISED-ROBUST-IMAGE-WATERMARKING
---

**Samples with extreme compound noises:**
---
![Untitled presentation](https://user-images.githubusercontent.com/44358874/221333997-b3393ac5-6a41-4794-8712-b14b04ec46d0.png)
![Untitled presentation (1)](https://user-images.githubusercontent.com/44358874/221334009-2b701332-d208-4fc0-94db-220a3e0613f7.png)
We illustrate the robustness of our proposed scheme by showing a few examples of extremely distorted cover images from Imagenet with their corresponding recovered watermarks.

**Noise tolerance study:**
---
![noise_tolerance](https://user-images.githubusercontent.com/44358874/221332916-a5b70bc7-ec7d-41d9-bf85-45c08fe148f9.jpg)
We perform an experiment to test the tolerance of our proposed scheme against increasing levels of trained and untrained noises. As expected the performance decreases steadily with an increase in the degree of noise.

**Triplet loss:**
---
![triplet](https://user-images.githubusercontent.com/44358874/218556589-7aed4be5-b82b-4d96-a9f7-bf0b8ac7e2eb.png)

A diagram illustrating the functionality of triplet loss. The anchor and the positive image are brought closer via a distance metric and the negative image is pushed away. This helps the model learn similar features between the first pair while not collapsing into a local minima.

**Watermark generation example:**
---
![watermark_generation](https://user-images.githubusercontent.com/44358874/218556635-882cd8b2-7461-4dc0-b494-ec4ee3d0ba36.png)
(Left) A sample 128x128x3 cover image from our subset of the imagenet validation set.
(Right) The watermark generated by isolating the first channel and binarizing the pixels which range from 0 to 255, to 0 or 1, based on the threshold 128 (half of 255).

<!-- 
**Simple overview:**
---
![ae_watermarking](https://user-images.githubusercontent.com/44358874/218557441-584e464a-94fe-4cd0-bc4f-46789cb10152.png)

This is a simple overview of the modern autoencoder style watermarking scheme. -->

<!-- **Training noises:**
---
![training_noise](https://user-images.githubusercontent.com/44358874/218556477-2f40b883-203f-484d-8eda-67cc0ecdfbea.jpg)
The noises used for the training of our scheme.

**Testing noises:**
---
![testing_noise](https://user-images.githubusercontent.com/44358874/218556552-2b62555d-2ec8-44b8-a80f-52950190d64e.jpg)
The noises selected for the testing of our scheme, to check for robustness against untrained noises. -->
