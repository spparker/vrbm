#include "matrix.h"

Matrix::Matrix(int x, int y, float* data)
{
	dimX = x;
	dimY= y;
	vals = data;
}

Matrix::Matrix(int x, int y)
{
	dimX = x;
	dimY = y;
	vals = (float*)malloc(x*y*sizeof(float));
	for(int i=0;i<x*y;i++)
		vals[i]=0;
}

float Matrix::getElem(int idx)
{
	if (idx<dimX*dimY)
		return vals[idx];

	else
		cout<<"Invalid Index Reference!"<<endl;

}

void Matrix::setLayer(float* data)
{
	vals = data;
}

void Matrix::print()
{
	for (int i=0;i<dimY;i++)
	{
		for(int j=0;j<dimX;j++)
		{
			cout<<"["<<vals[i*dimX + j]<<"]";
		}
		cout<<endl;
	}
}
