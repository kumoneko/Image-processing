#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cv.h>
#include <highgui.h>
//#include <stdio.h>
using namespace cv;
using namespace std;


void drawHistImg(const Mat &src, Mat &dst) {
	int histSize = 256;
	float histMaxValue = 0;
	for (int i = 0; i<histSize; i++) {
		float tempValue = src.at<float>(i);
		if (histMaxValue < tempValue) {
			histMaxValue = tempValue;
		}
	}

	float scale = (0.9 * 256) / histMaxValue;
	for (int i = 0; i<histSize; i++) {
		int intensity = static_cast<int>(src.at<float>(i)*scale);
		line(dst, Point(i, 255), Point(i, 255 - intensity), Scalar(0));
	}
}

Mat sobel(Mat img) {
	Mat src = img;
	GaussianBlur(src, src, Size(3, 3), 0, 0);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U
	Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	Mat dst1, dst2;
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
	threshold(dst1, dst2, 80, 255, THRESH_BINARY | THRESH_OTSU);
	return dst2;
}

Mat erode(Mat img) {
	Mat src = img;
	Mat src2;
	threshold(src, src2, 120, 255, THRESH_BINARY);
	Mat dst1;
	Mat dst2;
	Mat dst3;
	erode(src2, dst1, Mat());
	return dst1;
}

Mat negative(Mat img) {

	for (int i=0;i<img.rows;i++) {
		for (int j = 0;j<img.cols;j++) {
			
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}

	return img;
}

Mat coloradd(Mat img) {

	for (int i = 0; i<img.rows; i++) {
		for (int j = 0; j<img.cols; j++) {


			if (img.at<uchar>(i, j) > 50) {
				img.at<uchar>(i, j) = img.at<uchar>(i, j)-50;
			}
			else {
				img.at<uchar>(i, j) = 0;
			}
			
		}
	}
	return img;
}

Mat doubleselect(Mat img,int a,int b,int n) {
	
	for (int i = 0; i<img.rows; i++) {
		for (int j = 0; j<img.cols; j++) {
			if (img.at<uchar>(i, j) < b&&img.at<uchar>(i, j) > a) {
				img.at<uchar>(i, j) = n;
			}
			else { 
				img.at<uchar>(i, j) = 255; 
			}
		}
	}
	return img;
}

void Sobel(string BMP) {

	Mat src = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur(src, src, Size(3, 3), 0, 0);
//	doubleselect(src, 120, 200, 0);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  //CV_8U
	Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	Mat dst1, dst2;
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst1);
	
//	dst1 = negative(dst1);
	dst2 = dst1.clone();
	
	
	imshow("oring",src);
	
	imshow("Sobel", dst1);
	imshow("Sobel_2", dst2);
	
}

void Canny(string BMP) {

	Mat src = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur(src, src, Size(2, 2), 0, 0);
	Mat dst1, dst2;
	Canny(src, dst1, 100, 150, 3);
	threshold(dst1, dst2, 128, 255, THRESH_BINARY_INV);  //反轉影像，讓邊緣呈現黑線
	imshow("origin", src);
	imshow("Canny", dst1);
	imshow("Canny_2", dst2);
	



}

void Laplacian(string BMP) {
	Mat src = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur(src, src, Size(3, 3), 0, 0);
	Mat dst1, dst2, dst3;
	Laplacian(src, dst1, CV_16S, 3, 3, 0, BORDER_DEFAULT);
	convertScaleAbs(dst1, dst2);  //轉成CV_8U

	threshold(dst2, dst3, 80, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("origin", src);
	imshow("Laplacian", dst2);
	imshow("Laplacian_2", dst3);
	waitKey(0);

}

void Erode(string BMP) {
	Mat src = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2;
	threshold(src, src2, 120, 255, THRESH_BINARY);
	Mat dst1;
	Mat dst2;
	Mat dst3;
	erode(src2, dst1, Mat());
	
	Mat erodeStruct = getStructuringElement(MORPH_RECT, Size(5, 5));
	erode(src2, dst3, erodeStruct);
	
	imshow("erode", dst1);
	//imwrite("erodeLIN.jpg", dst1);
	imshow("erode2", dst3);
}

void Dilate(string BMP) {
	Mat src = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	Mat src2;
	threshold(src, src2, 120, 255, THRESH_BINARY);
	Mat dst2;
	dilate(src2, dst2, Mat());
	imshow("dilate", dst2);
}

void Contours(string BMP) {
	Mat src = imread(BMP, CV_LOAD_IMAGE_COLOR);
	Mat src_gray = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	Mat contoursImg = src.clone();

	Mat edge;
	blur(src_gray, src_gray, Size(3, 3));
	Canny(src_gray, edge, 50, 150, 3);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	findContours(edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i<contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), 255);
		drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
	}

	imshow("origin", src);
	imshow("result", contoursImg);
}

void threshold(string BMP) {
	Mat src = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = src.clone();
	Mat img2,img3;
	img3 = src.clone();
	img = erode(img);

	int i, j;
	imshow("org", src);
	for (i = 0;i < src.rows ;i++) {
		for (j = 0;j < src.cols ;j++) {
			
			if (src.at<uchar>(i,j) < 150 && src.at<uchar>(i, j) > 140) {
				src.at<uchar>(i, j) = 255;
			}
			else src.at<uchar>(i, j) = 0;
		
		}
	}
	addWeighted(img, 0.5, src, 0.5, 0, img2);
	
	for (i = 0; i < img2.rows; i++) {
		for (j = 0; j < img2.cols; j++) {
			if (img2.at<uchar>(i, j) == 255)img2.at<uchar>(i, j) = 0;
			if (img2.at<uchar>(i, j) < 129 && img2.at<uchar>(i, j) > 125) {
				img2.at<uchar>(i, j) = 255;
			}
		}
	}
	
	addWeighted(img3, 0.5, img2, 0.5, 0, img2);

	threshold(img2, img2, 210, 255, THRESH_BINARY);

	imshow("threshold",src);

	imshow("erode", img);

	imshow("mystyle",img2);


}

void gaussian(string BMP) {
	Mat img = imread(BMP, CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst;
	GaussianBlur(img, dst, Size(6, 6), 0, 0);
	doubleselect(dst,130,250,0);
	imshow("Gaussian",dst);
//	imwrite("T1d.jpg",dst);
}

void laplacian(string BMP) {
	Mat src = imread(BMP);
     Mat sharpened;
	 Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
     filter2D(src, sharpened, src.depth(), kernel);

	 imshow("or", src);
     imshow("sharpened", sharpened);
	 waitKey();

}

void binary(string img) {
	Mat src = imread(img, CV_LOAD_IMAGE_GRAYSCALE);
	Mat pic = src.clone();
	threshold(src, pic, 150, 255, THRESH_BINARY);
	imshow("origin",src);
	imshow("THRESHOLD",pic);
	
}

void hist(string img) {
	Mat src = imread(img, CV_LOAD_IMAGE_GRAYSCALE);
	int histSize = 256;
	float range[] = { 10, 245 };    //X軸的範圍 0~255
	const float* histRange = { range };
	Mat histImg;
	calcHist(&src, 1, 0, Mat(), histImg, 1, &histSize, &histRange);

	Mat showHistImg(256, 256, CV_8UC1, Scalar(255));  //把直方圖秀在一個256*256大的影像上
	drawHistImg(histImg, showHistImg);
	imshow("window1", src);
	threshold(showHistImg, showHistImg, 128, 255, THRESH_BINARY_INV);
	namedWindow("windows2", WINDOW_NORMAL);
	imshow("windows2", showHistImg);

}

void black(string img) {
	Mat src = imread(img, -1);
	Mat pic = src.clone();
	
	
	
	
	for (int i = 0; i<pic.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			pic.at<uchar>(i, j) = 255 - pic.at<uchar>(i, j);
		}
	}



	imshow("origin", src);
	imshow("negative", pic);

}

void remap(string img) {
	Mat src = imread(img);
	Mat map_x_1, map_y_1, map_x_2, map_y_2, map_x_3, map_y_3;
	map_x_1.create(src.size(), CV_32FC1);
	map_y_1.create(src.size(), CV_32FC1);
	map_x_2.create(src.size(), CV_32FC1);
	map_y_2.create(src.size(), CV_32FC1);
	map_x_3.create(src.size(), CV_32FC1);
	map_y_3.create(src.size(), CV_32FC1);

	for (int iH = 0; iH<src.rows; iH++) {
		for (int iW = 0; iW<src.cols; iW++) {
			map_x_1.at<float>(iH, iW) = iW;
			map_y_1.at<float>(iH, iW) = src.rows - iH;

			map_x_2.at<float>(iH, iW) = src.cols - iW;
			map_y_2.at<float>(iH, iW) = iH;

			map_x_3.at<float>(iH, iW) = src.cols - iW;
			map_y_3.at<float>(iH, iW) = src.rows - iH;
		}
	}

	Mat dst1, dst2, dst3;
	remap(src, dst1, map_x_1, map_y_1, CV_INTER_LINEAR);
	remap(src, dst2, map_x_2, map_y_2, CV_INTER_LINEAR);
	remap(src, dst3, map_x_3, map_y_3, CV_INTER_LINEAR);

	imshow("origin", src);
//	imshow("remap_1", dst1);
	imshow("remap", dst2);
//	imshow("remap_3", dst3);
	waitKey(0);
}

void LogEnhance(IplImage* img, IplImage* dst)
{
	// 由于oldPixel:[1,256],则可以先保存一个查找表  
	uchar lut[256] = { 0 };

	double temp = 255 / log(256);

	for (int i = 0; i<255; i++)
	{
		lut[i] = (uchar)(temp* log(i + 1) + 0.5);
	}

	for (int row = 0; row <img->height; row++)
	{
		uchar *data = (uchar*)img->imageData + row* img->widthStep;
		uchar *dstData = (uchar*)dst->imageData + row* dst->widthStep;

		for (int col = 0; col<img->width; col++)
		{
			for (int k = 0; k<img->nChannels; k++)
			{
				uchar t1 = data[col*img->nChannels + k];
				dstData[col*img->nChannels + k] = lut[t1];
			}
		}
	}
}


int main() {
	
	string name;


	const char* Path = "chicken.jpg";
	IplImage *img = cvLoadImage(Path, CV_LOAD_IMAGE_ANYCOLOR);
	IplImage *dst = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
	LogEnhance(img,dst);
	cvShowImage("SRC", img);
	cvShowImage("DST", dst);
	
	
	name = "chicken.jpg";

	Sobel(name);
	Canny(name);
	Laplacian(name);
	Erode(name);
	Dilate(name);
	Contours(name);
	threshold(name);
	binary(name);
	gaussian(name);
	laplacian(name);
	hist(name);
//	black(name);
	remap(name);


	waitKey(0);

	return 0;
}