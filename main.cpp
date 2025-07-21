// Block Matching Algorithm for Stereo Matching
//

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef vector<int> Table1;
typedef vector<Table1> Table2;
typedef vector<Table2> Table3;

Mat leftImage, rightImage, disparityMap, disparityImage;
Table3 initialCost, aggregatedCost;
int width, height, windowSize, levels;

int computeInitialCost(int x, int y, int label);
int computeAggregatedCost(int x, int y, int label);
int findBestAssignment(int x, int y);

int main()
{
	windowSize = 5;
	levels = 16;

	// Start timer
	auto start = chrono::steady_clock::now();

	// Read stereo image
	leftImage = imread("left.png", IMREAD_GRAYSCALE);
	rightImage = imread("right.png", IMREAD_GRAYSCALE);

	// Use gaussian filter
	GaussianBlur(leftImage, leftImage, Size(5, 5), 0.68);
	GaussianBlur(rightImage, rightImage, Size(5, 5), 0.68);

	// Get image size
	width = leftImage.cols;
	height = leftImage.rows;

	// Cache initial matching cost
	initialCost = Table3(height, Table2(width, Table1(levels)));
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			for (int i = 0; i < levels; i++)
				initialCost[y][x][i] = computeInitialCost(x, y, i);

	// Cache aggregated matching cost
	aggregatedCost = Table3(height, Table2(width, Table1(levels)));
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			for (int i = 0; i < levels; i++)
				aggregatedCost[y][x][i] = computeAggregatedCost(x, y, i);

	// Initialize disparity map
	disparityMap = Mat::zeros(height, width, CV_8U);

	// Update disparity map
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int label = findBestAssignment(x, y);
			disparityMap.at<uchar>(y, x) = label;
		}

	// Update disparity image
	int scaleFactor = 256 / levels;
	disparityMap.convertTo(disparityImage, CV_8U, scaleFactor);

	// Show disparity image
	namedWindow("Disparity Image", WINDOW_NORMAL);
	imshow("Disparity Image", disparityImage);
	waitKey(1);

	// Save disparity image
	bool flag = imwrite("disparity.png", disparityImage);

	// Stop timer
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "\nRunning Time: " << chrono::duration<double, milli>(diff).count() << " ms" << endl;

	waitKey(0);

	return 0;
}

int computeInitialCost(int x, int y, int label)
{
	int leftPixel = leftImage.at<uchar>(y, x);
	int rightPixel = (x >= label) ? rightImage.at<uchar>(y, x - label) : 0;
	int cost = abs(leftPixel - rightPixel);

	return cost;
}

int computeAggregatedCost(int x, int y, int label)
{
	int cost = 0;
	for (int dy = -windowSize / 2; dy < windowSize / 2 + windowSize % 2; dy++)
		for (int dx = -windowSize / 2; dx < windowSize / 2 + windowSize % 2; dx++)
			cost += (y + dy >= 0 && y + dy < height&& x + dx >= 0 && x + dx < width) ? initialCost[y + dy][x + dx][label] : 0;

	return cost;
}

int findBestAssignment(int x, int y)
{
	int label, min = INT_MAX;
	for (int i = 0; i < levels; i++)
	{
		int cost = aggregatedCost[y][x][i];
		if (cost < min)
		{
			label = i;
			min = cost;
		}
	}

	return label;
}
