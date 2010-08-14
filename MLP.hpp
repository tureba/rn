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

#if (!defined(MLP_CLASS))
	#warning Do not include MLP.hpp directly. Instead, include MLP.
	#warning This will be done now, but you still need to fix your code.
	#include "MLP"
#elif !defined(MLP_HPP)

#define MLP_HPP

#include <cstring>

template <typename T>
MLP<T>::MLP (int _num_layers, int *_size_layers, T _learning_tax, T _learning_momentum)
{
	init(_num_layers, _size_layers, _learning_tax, _learning_momentum);
}


template <typename T>
MLP<T>::MLP (const char *filename)
{
	load(filename);
}


template <typename T>
MLP<T>::MLP (const MLP<T> &NN)
{
	copy(NN);
}


template <typename T>
void MLP<T>::init (int _num_layers, int *_size_layers, T _learning_tax, T _learning_momentum)
{
	num_layers = _num_layers;
	size_layers = new int[num_layers + 1];
	memcpy(size_layers, _size_layers, (num_layers + 1) * sizeof(int));

	learning_tax = _learning_tax;
	learning_momentum = _learning_momentum;

	num_training_sets = 0;
	training_sets = NULL;
}


template <typename T>
void MLP<T>::clean ()
{
}


template <typename T>
void MLP<T>::load (const char *filename)
{
}


template <typename T>
void MLP<T>::copy (const MLP<T> &NN)
{
}


template <typename T>
float MLP<T>::learn (T *input, T *output)
{
	return .0f;
}


template <typename T>
float MLP<T>::learn (const char *input, const char *output)
{
	return .0f;
}


template <typename T>
float MLP<T>::learn (int ifd, int ofd)
{
	return .0f;
}


template <typename T>
void MLP<T>::execute (T *input, T *output)
{

}


template <typename T>
void MLP<T>::execute (const char *input, const char *output)
{

}


template <typename T>
void MLP<T>::execute (int ifd)
{

}

#endif

/* vim: set syntax=cpp ts=8: */
