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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CONVERGED 0.0001		//Define quando uma rede converiu (erro pequeno d+)
#define PROB_SHUFFLE 0.01		//Define se vai embaralhar o vetor de entradas pro treino ou nao

int verbose=0;			//Define se vai mostrar os printf na tela pedindo informacao ou nao:
				//>10 mostra tudo
				//10<x<1 mostra somente as entradas depois que a rede ja foi treinada
				//0 nao mostra nada

/***************************************************************************************************
 Funcoes auxiliares
***************************************************************************************************/

/*
 Funcao que faz o MALLOC, mas faz uma checagem para dar erro na alocacao se houver.
*/
static void * xmalloc(size_t size)
{
	void *ptr = malloc(size);
	if (ptr == NULL) {
		fprintf(stderr, "erro ao alocar %d bytes - saindo\n", size);
		exit(100);
	}
	return ptr;
}

/*
 Funcao que embaralha um vetor de acordo com uma probabilidade PROB_SHUFFLE
*/
void shuffle(int **vector, int n){
	int i=0;
	int p1=0,p2=0;
	int tmp;

	//Verificando a probabilidade de embaralhar
	if ( (((double) random())/((double) RAND_MAX))>PROB_SHUFFLE) return;

	//Embaralhando
	for (i=0;i<n;i++){

		//Seleciona dois individuos aleatoriamente para trocar a posicao
		p1=(int)(((double) random())/((double) RAND_MAX)*n);
		p2=(int)(((double) random())/((double) RAND_MAX)*n);

		//Troca
		tmp=(*vector)[p1];
		(*vector)[p1]=(*vector)[p2];
		(*vector)[p2]=tmp;
	}
}


/***************************************************************************************************
 Estruturas da RN
***************************************************************************************************/
typedef double (*act_function)(double x);	//Função de ativação a ser utilizada
typedef double (*dev_act_function)(double x);	//Derivada da funcao de ativacao

typedef struct _MLP{
	int QT_LAYERS;	//Qt de camadas (Saida e' uma camada - Entrada nao e')
	int *QT_NEU;	//Qt de neuronios por camada
	int QT_INPUT;	//Qt de entradas
	double lt;	//Learning Tax (Taxa de Aprendizagem)
	double **valueOfNeuron_noF;
	double ***W;	//Ponteiro para as matrizes de pesos - Pesos associados aos neuronios da esquerda e direita
			//Lembrando que W0 sera' a matriz de pesos associados com os dados da entrada com a primeira camada
			//Lembrando tambem que havera 1 W[i][QT_NEU][k] associado ao BIAS
			//Exemplo:
			//QT_NEU={3 , 4 , 2}
			// W1=[]4*4
			// W2=[]5*2
	act_function f;
	dev_act_function dev_f;
} MLP;


/***************************************************************************************************
 Funcoes da RN: Inicializacao, Execucao e Aprendizagem
***************************************************************************************************/

/*
 Funcao que inicializa as VARIAVEIS/Vetores/Matrizes da REDE-NEURAL.
 --
 QT_NEU sera criado
 *W sera criado - Valores aleatorios para wij
*/
MLP * start_vars(int QT_LAYERS, int *QT_NEU, int QT_INPUT, double learning_rate, act_function f, dev_act_function dev_f)
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
	nova_RN->QT_NEU = xmalloc(sizeof(double) * QT_LAYERS);
	memcpy(nova_RN->QT_NEU, QT_NEU, sizeof(double) * QT_LAYERS);

	//Iniciando a matriz com os valores de saida (sem a funcao de ativacao) de cada neuronio
	nova_RN->valueOfNeuron_noF=(double**) xmalloc(sizeof (double*) * nova_RN->QT_LAYERS);

	srand(time(NULL));
	// alocação das matrizes de peso
	nova_RN->W = (double ***) xmalloc(sizeof(double **) * QT_LAYERS);
	for (i=0; i < QT_LAYERS; i++) {

		//Iniciando a matriz com os valores de saida (sem a funcao de ativacao) de cada neuronio		
		nova_RN->valueOfNeuron_noF[i]=(double*) xmalloc(sizeof(double)*nova_RN->QT_NEU[i]);		

		// tam = QT_NEU[i] * ((i == 0) ? QT_INPUT : QT_NEU[i-1]);
		nova_RN->W[i] = (double **) xmalloc(sizeof(double) * (((i == 0) ? QT_INPUT : QT_NEU[i-1])+1));	//+1 por causa do BIAS

		// inicialização com valores aleatórios entre 0 e 1
		for (j = 0; j < ((i == 0) ? QT_INPUT : QT_NEU[i-1])+1; j++) {
			nova_RN->W[i][j] = (double *) xmalloc(sizeof(double) * QT_NEU[i]);
			for (k = 0; k < QT_NEU[i]; k++)
				nova_RN->W[i][j][k] = (((double) random())/((double) RAND_MAX)-0.5)/100;
		}
	}

	return nova_RN;
}

/*
 Funcao que "Executa" a rede-neural com a entrada (input)
e coloca na saida (output) os resultados obtidos.
*/
void execute(MLP *RN, double *input, double *output){
	int i,j,k;

	//Valores associados a cada neuronio da primeira camada escondida.
	double *result_values;
	double *tmp_values;
	result_values = (double*)xmalloc( sizeof(double) * RN->QT_NEU[0] );
	
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
	memcpy(RN->valueOfNeuron_noF[0],result_values,(sizeof(double)*RN->QT_NEU[0]));

	//Aplicando FUNCAO DE ATIVACAO
	for (j=0;j<RN->QT_NEU[0];j++) result_values[j]=RN->f(result_values[j]);

	//=====================================================
	//Todas as camadas, lembrando que entrada nao é camada
	for (k=1;k<RN->QT_LAYERS;k++){
		tmp_values=(double*)xmalloc( sizeof(double) * RN->QT_NEU[k] );

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
		memcpy(RN->valueOfNeuron_noF[k],tmp_values,(sizeof(double)*RN->QT_NEU[k]));

		//Aplicando FUNCAO DE ATIVACAO
		for (j=0;j<RN->QT_NEU[k];j++) tmp_values[j]=RN->f(tmp_values[j]);

		//Resultado da camada anterior, e' o que calculamos agora
		free(result_values);
		result_values = tmp_values;
	}

	//Retorna valores encontrados para a ultima camada
	memcpy(output,result_values, (sizeof(double)*RN->QT_NEU[k-1]) );
	free(result_values);
}

/*
 Funcao que "Aprende", ou seja, dado um vetor de entradas (input) e um vetor de saidas (output),
usa a matriz de pesos, faz as contas e verifica a saida, compara com a saida desejada e ja faz o
back propagation.
 Retorna o somatorio do erro ou coisa assim
*/
double learn(MLP *RN, double *input, double *output){

	double erro;	

	double *current_output;		//Saida atual da rede neural
	int i,j,k;
	double xi;
	current_output= xmalloc(sizeof(double)*RN->QT_NEU[RN->QT_LAYERS-1]);

	//Obtendo a saida atual da rede neural, obtendendo tambem o RN->valueOfNeuron_noF
	execute(RN,input,current_output);	


	// ===================================================================
	// ----==== Fazendo os ajustes dos pesos para a ULTIMA CAMADA ====----
	// ===================================================================
	double *ej_old;
	double *ej;
	ej=(double*) xmalloc(sizeof(double)*RN->QT_NEU[RN->QT_LAYERS-1]);

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
		ej_old=(double *) xmalloc(sizeof(double)*RN->QT_NEU[i+1]);
		memcpy(ej_old, ej, (sizeof(double)*RN->QT_NEU[i+1]) );

		free(ej);

		ej=(double*) xmalloc(sizeof(double)*RN->QT_NEU[i]);

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
 Funcoes de ativacao
***************************************************************************************************/
double sigmoid(double x){
	return 1/(1+exp(-x));
}
double dev_sigmoid(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

double tanh(double x){
	return (exp(x*2.0)-1.0)/(exp(x*2.0)+1);
}

double dev_tanh(double x){
	return 1.0 - pow(tanh(x),2.0);
}

double line(double x){
	return x;
}
double dev_line(double x){
	printf("LINE nao pode ser usada (derivada).");
	exit(101);
	return 1;
}

/***************************************************************************************************
 Funcoes de interface e chamada continua de treinamento
***************************************************************************************************/

/*
 Funcao que ira receber os dados de entrada da tela e
fazer o treinamento da RN ate que convirja ou atinja um
numero maximo de iteracoes
*/
void receive_inputs_and_training(MLP **RN){
	int i,j=0;
	int nLayers=0;		//Qtd camadas
	int nInput=0;		//Qtd entrada (tamanho vetor input[])
	int nTraining=0;	//Qtd de entrada para treinamento
	int nOutput=0;		//Qtd de neuronios da camada de saida
	int MAXITERATION;	//Nro maximo de iteracoes (caso nao convirja)
	int *shuffleidx;	//Vetor que contera os indices do input[] e output[], usado para embaralhar
	double lr;		//Taxa de aprendizagem
	double **trainingInput,**trainingOutput;
	int *qt_neu;

	//Recebendo os dados de entrada
	if (verbose>10) printf("Taxa de Aprendizagem: ");
	scanf(" %lf",&lr);
	if (verbose>10) printf("Digite o numero de entradas (tamanho do vetor input) da MLP: ");
	scanf(" %i",&nInput);
	if (verbose>10) printf("Digite o numero de camadas da MLP: ");
	scanf(" %i",&nLayers);
	qt_neu=(int*) xmalloc(sizeof(int)*(nLayers+1));
	for (i=0;i<nLayers;i++){
		if (verbose>10) printf("Digite o numero de neuronios da camada %i (Entrada nao e' camada. Saida e'.): ",i+1);
		scanf(" %i",&qt_neu[i]);
		if (i==nLayers-1) nOutput=qt_neu[i];
	}

	//CRIANDO A REDE NEURAL!!!
	*RN = start_vars(nLayers, qt_neu, nInput, lr, sigmoid, dev_sigmoid);
	if (verbose>10) printf("Quantidade de entrada para treinamento (Qtd de individuos para treinar):");
	scanf(" %i",&nTraining);

	//Alocando espaco para os valores de entrada e saida
	trainingInput = (double **) xmalloc(sizeof(double *)*nTraining);
	trainingOutput= (double **) xmalloc(sizeof(double *)*nTraining);

	//Alocando espaco para o embaralha
	shuffleidx=(int*) xmalloc(sizeof(int)*nTraining);

	//Recebendo as entradas e saidas para treinamento
	for (i=0;i<nTraining;i++){
		shuffleidx[i]=i;
		trainingInput [i]=(double*) xmalloc(sizeof(double) * nInput);
		trainingOutput[i]=(double*) xmalloc(sizeof(double) * nOutput);
		for (j=0;j<nInput;j++){
			if (verbose>10) printf("Individuo %i, Entrada %i: ",i+1,j+1);
			scanf(" %lf",&trainingInput[i][j]);
		}
		for (j=0;j<nOutput;j++){
			if (verbose>10) printf("Individuo %i, Saida %i: ",i+1,j+1);
			scanf(" %lf",&trainingOutput[i][j]);
		}
	}

	//Recebendo nro maximo de iteracoes
	if (verbose>10) printf("Numero maximo de iteracoes: ");
	scanf(" %i",&MAXITERATION);

	//Preparando-se para treinar
	if (verbose) printf("\nTreinando, aguarde...\n");	

	fflush(stdout);	

	double erro=CONVERGED+1;
	i=0;
	int percent[10]={0};		//Usado para mostrar uma porcentagem de concluido na tela

	//Treinando!!!
	while (erro>CONVERGED && i<MAXITERATION){
		i++;

		//Mostrar uma porcentagem de concluido na tela
		if (verbose){
			if ((double)((double)i/(double)MAXITERATION)>0.1 && !percent[0]){
				percent[0]=1;
				printf("10%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.2 && !percent[1]){
				percent[1]=1;
				printf("20%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.3 && !percent[2]){
				percent[2]=1;
				printf("30%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.4 && !percent[3]){
				percent[3]=1;
				printf("40%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.5 && !percent[4]){
				percent[4]=1;
				printf("50%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.6 && !percent[5]){
				percent[5]=1;
				printf("60%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.7 && !percent[6]){
				percent[6]=1;
				printf("70%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.8 && !percent[7]){
				percent[7]=1;
				printf("80%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.9 && !percent[8]){
				percent[8]=1;
				printf("90%%...");
			}
			if ((double)((double)i/(double)MAXITERATION)>0.999 && !percent[9]){
				percent[9]=1;
				printf("100%%\n");
			}
			fflush(stdout);
		}

		//Fazendo o embaralhamento
		erro=0;
		shuffle(&shuffleidx,nTraining);

		//Para cada individuo de entrada faz o treinamento
		for (j=0;j<nTraining;j++){
			erro+=learn(*RN, trainingInput[shuffleidx[j]], trainingOutput[shuffleidx[j]]);
		}
	}

	//Limpando memoria
	free(qt_neu);
	free(shuffleidx);
	free(trainingInput);
	free(trainingOutput);

	//Convergiu ou terminou? Mostra o erro.
	if (verbose) if(erro<=CONVERGED) printf("Rede Convergiu...\n");
	if (verbose) printf("Rede Treinada!!! (erro=%lf)\n",erro);
}

/*
 Funcao que recebe dados de entrada depois da rede ja treinada
e mostra a saida que a rede gera.
 - Veja que aqui a rede ja deve estar treinada
*/
void execute_trained(MLP *RN){
	int again=1;
	int i,j;
	double *input,*output;

	//Loop ate que usuario queira sair
	while (again){

		//Recebendo valores de entrada
		if (verbose) printf("Recebendo novo individuo:\n");
		input=(double*) xmalloc(sizeof(double) * RN->QT_INPUT);
		for (i=0;i<RN->QT_INPUT;i++){
			if (verbose) printf("Entrada %i: ",i+1);
			fflush(stdout);
			scanf(" %lf", &input[i]);
		}

		//Executando a rede neural
		output=(double *) xmalloc(sizeof(double) * RN->QT_NEU[RN->QT_LAYERS-1]);
		execute(RN,input,output);

		//Mostrando as saidas que a Rede Neural ja obteve.
		if (verbose) printf("Saida(s):\n");
		for (i=0;i<RN->QT_NEU[RN->QT_LAYERS-1];i++){
			printf("%lf ",output[i]);
			fflush(stdout);
		}

		free(input);
		free(output);

		//Sair? Ou de novo?
		if (verbose) printf("\nOutro individuo (1/0): ");
		fflush(stdout);
		scanf(" %i",&again);
	}	
}

/***************************************************************************************************
 MAIN
***************************************************************************************************/
int main(int argc, char *argv[]){

	//Recebendo parametro de entrada (VERBOSE)
	if (argc==1) verbose=999;
	else verbose=atoi(argv[1]);	

	//Criando a rede MLP, Treinando e Executando
	MLP *myNN;
	receive_inputs_and_training(&myNN);
	execute_trained(myNN);

	//Good bye
	fflush(stdout);
	return 0;

}
