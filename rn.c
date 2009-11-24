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
#include <error.h>

#include "rn.h"


float max_error = 0.0001f;		//Define quando uma rede converiu (erro pequeno d+)
float prob_shuffle = 0.01f;		//Define se vai embaralhar o vetor de entradas pro treino ou nao
int max_iter = 10000;
int verbose = 0;



/***************************************************************************************************
 Funções auxiliares
***************************************************************************************************/


/*
 Função que faz o malloc, com uma checagem para retornar erro na alocação quando houver.
*/
void * xmalloc(size_t size)
{
	void *ptr = malloc(size);
	if ((ptr == NULL) && size)
		error(100, 0, "erro ao alocar %d bytes - saindo\n", size);
	return ptr;
}

/*
 Função que faz o realloc, com uma checagem para retornar erro na alocação quando houver.
*/
void * xrealloc(void *p, size_t size)
{
	void *ptr = realloc(p, size);
	if ((ptr == NULL) && size)
		error(100, 0, "erro ao alocar %d bytes - saindo\n", size);
	return ptr;
}

/*
 Função que embaralha um vetor de acordo com uma probabilidade PROB_SHUFFLE
*/
void shuffle(float **vector, int n)
{
	int i=0;
	int p1=0,p2=0;
	float tmp;

	//Verificando a probabilidade de embaralhar
	if (uniforme01 > prob_shuffle)
		return;

	//Embaralhando
	for (i=0;i<n;i++) {
		//Seleciona dois individuos aleatoriamente para trocar a posicao
		p1 = (int)(uniforme01 * n);
		p2 = (int)(uniforme01 * n);

		//Troca
		tmp = (*vector)[p1];
		(*vector)[p1] = (*vector)[p2];
		(*vector)[p2] = tmp;
	}
}


/***************************************************************************************************
 Funcoes da RN: Inicializacao, Execucao e Aprendizagem
***************************************************************************************************/

/*
 Funcao que inicializa as VARIAVEIS/Vetores/Matrizes da REDE-NEURAL.
 --
 QT_NEU sera criado
 *W sera criado - Valores aleatorios para wij
*/
MLP * start_vars(int QT_LAYERS, int *QT_NEU, int QT_INPUT, float learning_rate, act_function f, dev_act_function dev_f)
{
	int i,j, k;
	// alocação da estrutura principal
	MLP *nova_RN = xmalloc(sizeof(MLP));

	// atribuicao da taxa de aprendizagem
	nova_RN->lt = learning_rate;	

	// atribuição das funções da RN
	nova_RN->f = f;
	nova_RN->dev_f = dev_f;

	// atribuição dos tamanhos da RN
	nova_RN->QT_LAYERS = QT_LAYERS;
	nova_RN->QT_INPUT = QT_INPUT;

	// atribuição dos tamanhos de cada camada
	nova_RN->QT_NEU = xmalloc(sizeof(int) * QT_LAYERS);
	memcpy(nova_RN->QT_NEU, QT_NEU, sizeof(int) * QT_LAYERS);

	//Iniciando a matriz com os valores de saida (sem a funcao de ativacao) de cada neuronio
	nova_RN->valueOfNeuron_noF=(float **) xmalloc(sizeof (float *) * nova_RN->QT_LAYERS);

	srand(time(NULL));
	// alocação das matrizes de peso
	nova_RN->W = (float ***) xmalloc(sizeof(float **) * QT_LAYERS);
	for (i=0; i < QT_LAYERS; i++) {

		//Iniciando a matriz com os valores de saida (sem a funcao de ativacao) de cada neuronio		
		nova_RN->valueOfNeuron_noF[i] = (float *) xmalloc(sizeof(float) * nova_RN->QT_NEU[i]);		

		// tam = QT_NEU[i] * ((i == 0) ? QT_INPUT : QT_NEU[i-1]);
		nova_RN->W[i] = (float **) xmalloc(sizeof(float) * (((i == 0) ? QT_INPUT : QT_NEU[i-1])+1));	//+1 por causa do BIAS

		// inicialização com valores aleatórios entre 0 e 1
		for (j = 0; j < ((i == 0) ? QT_INPUT : QT_NEU[i-1])+1; j++) {
			nova_RN->W[i][j] = (float *) xmalloc(sizeof(float) * QT_NEU[i]);
			for (k = 0; k < QT_NEU[i]; k++)
				nova_RN->W[i][j][k] = (((float) rand())/((float) RAND_MAX)-0.5)/100;
		}
	}

	nova_RN->QT_TREINAMENTO = 0;
	nova_RN->TREINAMENTOS = NULL;

	return nova_RN;
}

/*
 Funcao que "Executa" a rede-neural com a entrada (input)
e coloca na saida (output) os resultados obtidos.
*/
void execute(MLP *RN, float *input, float *output){
	int i,j,k;

	//Valores associados a cada neuronio da primeira camada escondida.
	float *result_values;
	float *tmp_values;
	result_values = (float*)xmalloc( sizeof(float) * RN->QT_NEU[0] );
	
	//Inicializando
	for (j=0;j<RN->QT_NEU[0];j++) result_values[j]=0;

	//Processando dados da ENTRADA. tmp_values=sum(....
	for (i=0;i<RN->QT_INPUT+1;i++){		//+1 = BIAS
		for (j=0;j<RN->QT_NEU[0];j++){
			if (i==RN->QT_INPUT) 	result_values[j]+=1		*RN->W[0][i][j];	//BIAS
			else 			result_values[j]+=input[i]	*RN->W[0][i][j];
		}
	}

	//Setando os valores de saida de cada neuronio na matrizW_noF (sem aplicar a funcao de ativacao)
	memcpy(RN->valueOfNeuron_noF[0],result_values,(sizeof(float)*RN->QT_NEU[0]));

	//Aplicando FUNCAO DE ATIVACAO
	for (j=0;j<RN->QT_NEU[0];j++) result_values[j]=RN->f(result_values[j]);

	//=====================================================
	//Todas as camadas, lembrando que entrada nao é camada
	for (k=1;k<RN->QT_LAYERS;k++){
		tmp_values=(float*)xmalloc( sizeof(float) * RN->QT_NEU[k] );

		//Inicializando
		for (j=0;j<RN->QT_NEU[k];j++) tmp_values[j]=0;

		//Processando dados da camada anterior. tmp_values=sum(....
		for (i=0;i<RN->QT_NEU[k-1]+1;i++){
			for (j=0;j<RN->QT_NEU[k];j++){
				if (i==RN->QT_NEU[k-1]) tmp_values[j]+=1		*RN->W[k][i][j];	//BIAS
				else 			tmp_values[j]+=result_values[i]	*RN->W[k][i][j];
			}
		}
		//Setando os valores de saida de cada neuronio na matrizW_noF (sem aplicar a funcao de ativacao)
		memcpy(RN->valueOfNeuron_noF[k],tmp_values,(sizeof(float)*RN->QT_NEU[k]));

		//Aplicando FUNCAO DE ATIVACAO
		for (j=0;j<RN->QT_NEU[k];j++) tmp_values[j]=RN->f(tmp_values[j]);

		//Resultado da camada anterior, e' o que calculamos agora
		free(result_values);
		result_values = tmp_values;
	}

	//Retorna valores encontrados para a ultima camada
	memcpy(output,result_values, (sizeof(float)*RN->QT_NEU[k-1]) );
	free(result_values);
}

/*
 Funcao que "Aprende", ou seja, dado um vetor de entradas (input) e um vetor de saidas (output),
usa a matriz de pesos, faz as contas e verifica a saida, compara com a saida desejada e ja faz o
back propagation.
 Retorna o somatorio do erro ou coisa assim
*/
float learn(MLP *RN, float *input, float *output){

	float erro;	

	float *current_output;		//Saida atual da rede neural
	int i,j,k;
	float xi;
	current_output= xmalloc(sizeof(float)*RN->QT_NEU[RN->QT_LAYERS-1]);

	//Obtendo a saida atual da rede neural, obtendendo tambem o RN->valueOfNeuron_noF
	execute(RN,input,current_output);	


	// ===================================================================
	// ----==== Fazendo os ajustes dos pesos para a ULTIMA CAMADA ====----
	// ===================================================================
	float *ej_old;
	float *ej;
	ej=(float*) xmalloc(sizeof(float)*RN->QT_NEU[RN->QT_LAYERS-1]);

	//Percorrendo todos os neuronios da ultima camada
	for (j=0;j<RN->QT_NEU[RN->QT_LAYERS-1];j++){

		//Calculando o erro
		ej[j]=(output[j]-current_output[j])*RN->dev_f(RN->valueOfNeuron_noF[RN->QT_LAYERS-1][j]);

		//Se existe mais que 1 camada, entao a camada anterior esta no RN
		if (RN->QT_LAYERS>1){

			//Percorrendo todos os neuronios da camada anterior
			for (i=0;i<RN->QT_NEU[RN->QT_LAYERS-2]+1;i++){
				if (i==RN->QT_NEU[RN->QT_LAYERS-2])	//CALCULO PARA O BIAS
						xi=1;
				else		xi=RN->f(RN->valueOfNeuron_noF[RN->QT_LAYERS-2][i]);
	
				//Acertando os pesos dos neuronios da camada anterior associados a este neuronio (j)
				RN->W[RN->QT_LAYERS-1][i][j]+=RN->lt*ej[j]*xi;
			}

		//Se existe somente 1 camada, entao a camada anterior e' a entrada. i.e. dados estao no input[]
		} else {

			//Percorrendo todas as entradas (Tamanho do vetor input[])
			for (i=0;i<RN->QT_INPUT+1;i++){
				if (i==RN->QT_INPUT)			//CALCULO PARA O BIAS
						xi=1;
				else		xi=input[i];

				//Acertando os pesos das entradas associadas a este neuronio (j)
				RN->W[RN->QT_LAYERS-1][i][j]+=RN->lt*ej[j]*xi;
			}
		}
	}

	// =============================================================================
	// ----==== Fazendo os ajustes dos pesos para as CAMADAS INTERMEDIARIAS ====----
	// =============================================================================
	for (i=RN->QT_LAYERS-2;i>=0;i--){

		//Copiando vetor de erros da camada posterior (usaremos para achar o erro da camada anterior)
		ej_old=(float *) xmalloc(sizeof(float)*RN->QT_NEU[i+1]);
		memcpy(ej_old, ej, (sizeof(float)*RN->QT_NEU[i+1]) );

		free(ej);

		ej=(float*) xmalloc(sizeof(float)*RN->QT_NEU[i]);

		//Percorrendo todos os neuronios da camada intermediaria
		for (j=0;j<RN->QT_NEU[i];j++){
			ej[j]=0;

			//Percorrendo todos os neuronios da camada posterior, e calculando o erro (ej)
			for (k=0;k<RN->QT_NEU[i+1];k++){

				//Calculo do ej. Repare que usamos o erro da camada posterior (ej_old)
				ej[j]+=(ej_old[k]*RN->W[i+1][j][k])*RN->dev_f(RN->valueOfNeuron_noF[i][j]);

			}


			//Se existe mais camadas, entao a camada anterior esta no RN
			if (i>0){

				//Percorredo todos os neuronios da camada anterior
				for (k=0;k<RN->QT_NEU[i-1]+1;k++){

					
					if (k==RN->QT_NEU[i])	//CALCULO PARA O BIAS
						xi=1;
					else	xi=RN->f(RN->valueOfNeuron_noF[i-1][k]);

					//Atualizando pesos da camada anterior associado a este neuronio (j)
					RN->W[i][k][j]+=RN->lt*ej[j]*xi;


				}

			//Se esta e' a ultima camada intermediaria, entao a camada anterior e' a entrada. i.e. dados estao no vetor input[]
			} else {

				//Percorrendo toda a entrada (input[])
				for (k=0;k<RN->QT_INPUT+1;k++){
					if (k==RN->QT_INPUT) 	//CALCULO PARA O BIAS
						xi=1;
					else	xi=input[k];

					//Atualizando pesos da entrada associado a este neuronio (j)
					RN->W[i][k][j]+=RN->lt*ej[j]*xi;
				}
			}

		}

	}

	//Calculando o erro para retornar
	erro=0;
	for (j=0;j<RN->QT_NEU[RN->QT_LAYERS-1];j++)
		erro += (output[j] - current_output[j])*(output[j] - current_output[j]);

	//Desalocando memoria
	free(ej);
	free(current_output);

	return erro;
	
}


/***************************************************************************************************
 Funções de ativação
***************************************************************************************************/

float sigmoid (float x)
{
	return 1/(1+expf(-x));
}
float dev_sigmoid (float x)
{
	return sigmoid(x)*(1-sigmoid(x));
}

float ftanh (float x)
{
	return (expf(x*2.0)-1.0)/(expf(x*2.0)+1);
}

float dev_tanh (float x)
{
	return 1.0 - powf(ftanh(x),2.0);
}
