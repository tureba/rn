/*
	Artificial Neural Networks Library - MLP implementation

	Copyright (C) 2010 Arthur Nascimento <tureba@gmail.com>

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>

#include "MLP"

/* Activation function: sigmoid (float) */
template <>
float MLP<float>::sigmoid (float x)
{
	return 1 / (1 + expf(-x));
}

	
template <>
float MLP<float>::dev_sigmoid (float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}


/* Activation function: sigmoid (double) */
template <>
double MLP<double>::sigmoid (double x)
{
	return 1 / (1 + exp(-x));
}

	
template <>
double MLP<double>::dev_sigmoid (double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}


/* Activation function: sigmoid (long double) */
template <>
long double MLP<long double>::sigmoid (long double x)
{
	return 1 / (1 + expl(-x));
}

	
template <>
long double MLP<long double>::dev_sigmoid (long double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}


/* Activation function: hiperbolic tangent (float) */
template <>
float MLP<float>::tanh (float x)
{
	return (expf(x * 2.0) - 1.0) / (expf(x * 2.0) + 1);
}


template <>
float MLP<float>::dev_tanh (float x)
{
	return 1.0 - powf(tanh(x), 2.0);
}


/* Activation function: hiperbolic tangent (double) */
template <>
double MLP<double>::tanh (double x)
{
	return (exp(x * 2.0) - 1.0) / (exp(x * 2.0) + 1);
}


template <>
double MLP<double>::dev_tanh (double x)
{
	return 1.0 - pow(tanh(x), 2.0);
}


/* Activation function: hiperbolic tangent (long double) */
template <>
long double MLP<long double>::tanh (long double x)
{
	return (expl(x * 2.0) - 1.0) / (expl(x * 2.0) + 1);
}


template <>
long double MLP<long double>::dev_tanh (long double x)
{
	return 1.0 - powl(tanh(x), 2.0);
}

/* vim: set syntax=cpp ts=8: */
