#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

class Matrix
{
private:
	int dimX;
	int dimY;
	float* vals;
public:
	Matrix(int x, int y, float* data);
	Matrix(int x, int y);

	float getElem(int idx);
	void setLayer(float* data);
	void print();
};
#endif
