/***************************************
**    Trabalho de Redes Neurais
**      Implementacao de MLP
**
** ICMC/USP - 2009-10-30
**
** Arthur Nascimento	5634455
** Lucas Vendramin	5961586
** Rodrigo P. Zeli	5635011
***************************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/**************************************************************************************************/

extern float max_error;		//Define quando uma rede converiu (erro pequeno d+)

extern float prob_shuffle;		//Define se vai embaralhar o vetor de entradas pro treino ou nao

extern int max_iter;

extern int verbose;


/***************************************************************************************************
 Funções auxiliares
***************************************************************************************************/

void * xmalloc (size_t size);
void * xrealloc (void *p, size_t size);

void shuffle (float **vector, int n);

#define uniforme01 (((float) rand()) / ((float) RAND_MAX))

/***************************************************************************************************
 Estruturas da RN
***************************************************************************************************/

//para o futuro: substituir os dados float por vetores de 4 floats para fazer 4 operações por instrução
//typedef float pixel __attribute__((vector_size (4 * sizeof(float))));

typedef float (*act_function)(float x);	//Função de ativação a ser utilizada

typedef float (*dev_act_function)(float x);	//Derivada da funcao de ativacao

typedef struct _MLP{
	int QT_LAYERS;	//Qt de camadas (Saida e' uma camada - Entrada nao e')
	int *QT_NEU;	//Qt de neuronios por camada
	int QT_INPUT;	//Qt de entradas
	float lt;	//Learning Tax (Taxa de Aprendizagem)
	float **valueOfNeuron_noF;
	float ***W;	//Ponteiro para as matrizes de pesos - Pesos associados aos neuronios da esquerda e direita
			//Lembrando que W0 sera' a matriz de pesos associados com os dados da entrada com a primeira camada
			//Lembrando tambem que havera 1 W[i][QT_NEU][k] associado ao BIAS
			//Exemplo:
			//QT_NEU={3 , 4 , 2}
			// W1=[]4*4
			// W2=[]5*2
	int QT_TREINAMENTO;
	float **TREINAMENTOS;
	act_function f;
	dev_act_function dev_f;
} MLP;

typedef struct _BMP_header {
	uint8_t tag[2]; //must be 'BM'
	uint32_t file_size;
	uint32_t reserved;
	uint32_t data_offset; //start of the bitmap
	uint32_t header_size; 
	uint32_t image_width; //
	uint32_t image_height; //
	uint16_t color_planes; //must be 1
	uint16_t bits_per_pixel; //must be 32
	uint32_t compression; //must be 0
	uint32_t image_size;
	uint32_t horizontal_resolution;
	uint32_t vertical_resolution;
	uint32_t colors_in_pallete; //must be 0
	uint32_t important_colors;
} BMP_header;

/***************************************************************************************************
 Funcoes da RN: Inicializacao, Execucao e Aprendizagem
***************************************************************************************************/

MLP * start_vars (int QT_LAYERS, int *QT_NEU, int QT_INPUT, float learning_rate, act_function f, dev_act_function dev_f);

void execute (MLP *RN, float *input, float *output);

float learn (MLP *RN, float *input, float *output);


/***************************************************************************************************
 Funcoes de ativacao
***************************************************************************************************/

float sigmoid (float x);

float dev_sigmoid (float x);

float ftanh (float x);

float dev_tanh (float x);

