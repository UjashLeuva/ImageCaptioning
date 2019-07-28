# dl_at1b
DeepLearning - Fashion Images classification

Folder Structure

- root
  - external_mnist
  - external_deepFashion
  - data


## fashion mnist

link: https://github.com/zalandoresearch/fashion-mnist

![Fashion mnist](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

The data set includes 
- 60k training examples 
- 10 test exmaples 

must first clone the fashion-minst repo as a sub directory to get the load function and files 
`git https://github.com/zalandoresearch/fashion-mnist fashion_mnist`

## DeepFashion Data: Large-scale Fashion (DeepFashion) Database 

Link: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html 

I will use the DeepFaship Category data set to build a CCN for categorising clothing items. 

![Category and Attribute Prediction Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/attributes.jpg)

citations: 
```
@inproceedings{liuLQWTcvpr16DeepFashion,
 author = {Ziwei Liu and Ping Luo and Shi Qiu and Xiaogang Wang and Xiaoou Tang},
 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 month = June,
 year = {2016} 
 }
 ```

```
 @inproceedings{liuYLWTeccv16FashionLandmark,
 author = {Ziwei Liu and Sijie Yan and Ping Luo and Xiaogang Wang and Xiaoou Tang},
 title = {Fashion Landmark Detection in the Wild},
 booktitle = {European Conference on Computer Vision (ECCV)},
 month = October,
 year = {2016} 
 }
 ````