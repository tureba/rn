/*
	Artificial Neural Networks Library - test program

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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <error.h>

#include "MLP"

int verbose;

void mostra_help(const char *prog)
{
	printf("Uso:\n\
%s <RNA> [-v] [-e erro_máximo] [-i máximo_iterações] [-t entrada saida]\n\
     <RNA>: arquivo que irá guardar a rede neural treinada ou de onde será lida uma rede neural treinada\n\
        -h: mostra esta mensagem\n\
        -v: habilita mais mensagens durante a execução\n\
        -e: define o erro máximo que precisa ser atingido\n\
        -i: define o número máximo de iterações para o treinamento\n\
	-t: informa dois arquivos que contêm informações de treinamento\n\
\n\
", prog);
	exit(0);

}

int main (int argc, char **argv, char **envp)
{

	int num_camadas = 2;
	int tam_camadas[3] = {5, 10, 5};
	int max_iter = 1000;
	float max_error = 0.001f;

	if (argc < 2) {
		std::cerr << "É necessário informar ao menos o arquivo inicial da RN" << std::endl;
		mostra_help(argv[0]);
	}

	MLP<float> RN(num_camadas, tam_camadas, 0.2f, 0.8f);

	for (int i = 2; i < argc; i++) {
		if (argv[i][0] == '-')
			switch (argv[i][1]) {
				case 'h':
					mostra_help(argv[0]);
					break;
				case 'v':
					verbose = 1;
					break;

				case 'i':
					if (++i >= argc)
						error(24, 0, "é necessário informar o número máximo de iterações");
					if (sscanf(argv[i], "%d", &max_iter) != 1)
						error(3, 0, "informe um número para a quantidade máxima de iterações");
					RN.max_iter = max_iter;
					break;

				case 'e':
					if (++i >= argc)
						error(11, 0, "é necessário informar o erro máximo do treinamento");
					if (sscanf(argv[i], "%f", &max_error) != 1)
						error(13, 0, "o erro foi fornecido incorretamente");
					RN.max_error = max_error;
					break;

				case 't':
					if (i+2 >= argc)
						error(23, 0, "é necessário informar o conjunto de treinamento (dois arquivos)");
					RN.learn(argv[i+1], argv[i+2]);
					i += 2;
					break;

			}
		else
			std::cerr << "Argumento não reconhecido: " << argv[i] << std::endl;
	}

	RN.execute(0);

	return 0;
}
