# CSC420_A2
> **Note for Linux users:** if you're using Ubuntu, make sure you've installed the following packages if
> you haven't done so already:
>
>     sudo apt-get update
>     sudo apt install software-properties-common
>     sudo add-apt-repository ppa:deadsnakes/ppa
>     sudo apt install python3.8
>     sudo apt-get install python-opencv
>     sudo apt-get install cmake
>     sudo apt-get install gcc g++
>     sudo apt-get install python3-dev python3-numpy
>     sudo pip install pil


## Summary
In this assignment, I have implement funtions `Gaussian_Model`, `Gaussian_Blur`, `convolution`, `Sobel_Operation`, `cutColoum`, `getGreyImge`, `cutRow`, `mainCol`, `mainRow`,`main`,`eigenvalues`,`scatterplot`,`corner_detection`,`label`.

`Gaussian_Model`: Implement Gaussian distribution.
`Gaussian_Blur`: Use Gaussian_Model to blur the image.
`convolution`: Convolution between one kernel and image.
`Sobel_Operation`: Image Gradient.
`cutColoum`: Cut one path from coloums.
`getGreyImge`: Transfer RGB to one channel.
`cutRow`: Cut one path from rows.
`mainCol`: N cut for coloums.
`mainRow`: N cut for rows.
`main`: mainRow + mainCol.
`eigenvalues`: Return 2 array with eigenvalues for M.
`scatterplot`: Draw scatter graph.
`corner_detection`: Binary image for corners.
`label`: RGB image for corners.


You can find details in the report and `resize.py`, and `corner_detection.py`.
Comments in two python file will tell you my thoughts.

## Tests Procedures

### Use `python3 resize.py` to get three images resize, crop, seam of every image. If you want test other image, please change the path in the program. It may take a long time to get seam image (20 mins). 

### Use `python3 corner_detection.py` to get scatterplot, binary image and RGB image of corners for image ex5.jpg.

## Academic Honest
I, Xiang Chen,  promise all codes are written by myself.