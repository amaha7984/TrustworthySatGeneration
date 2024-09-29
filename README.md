# TrustworthySatGeneration

**Description:**  
Conditional generation of satellite images from maps with the capability to detect the generated images from ground truth images, ensuring robust defenses against adversarial usage and supporting ethical applications in geospatial data.

## Overview
We present an interactive generative AI web application for the conditional generation of satellite images from maps. This application allows users to generate realistic satellite imagery that accurately captures various features such as buildings, roads, green areas, and water bodies. The generation process is powered by a self-supervised contrastive learning-based Generative Adversarial Network (GAN) without the need for explicit supervision during training.

## Forensic Detection
To address potential vulnerabilities of generated images to adversarial attacks, we have integrated a deep learning-based forensic framework capable of distinguishing between real and generated satellite images. This component ensures ethical use and safeguards against adversarial misuse.

## Motivation
With the increasing use of generative AI in various fields, it is critical to ensure that the generated content, especially in geospatial applications, is trustworthy and secure. Uncontrolled use of synthetic images can lead to misinformation and manipulation. This project aims to provide a robust solution for generating satellite imagery that is transparent and safeguarded against adversarial exploitation.

## Useful For
- **Public Benefit Projects:** Generating visual representations of historic landmarks or recreating old geographical features to preserve cultural heritage.
- **Environmental Analysis:** Supporting efforts to track changes in land use, monitor environmental impact, and promote sustainable development.
- **Researchers** exploring secure and ethical applications of generative AI in geospatial domains.
- **Government and Regulatory Bodies** looking to monitor and regulate synthetic satellite imagery.
- **Developers** building applications for realistic map-to-satellite image translation.
- **Organizations** focused on mitigating adversarial attacks in remote sensing and geospatial data.


## Features
- **Interactive Satellite Image Generation:** Create realistic satellite images from map inputs.
- **Self-Supervised Learning:** Powered by contrastive learning-based GANs.
- **Forensic Detection:** Identify whether a satellite image is real or generated.
- **Email Alert System:** Sends notifications for detected generated images using the SendGrid library.

## Technical Details
- **Model Architecture:** Self-supervised contrastive learning GAN.
- **Forensic Module:** Deep learning framework for real vs. generated classification.
- **Notification:** Integrated with SendGrid for alert system.

## Conclusion
With these innovations, we present a trustworthy satellite image generation system from map inputs, supporting ethical and secure applications in geospatial data.


