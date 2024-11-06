#!/bin/bash

###########################################################################
###				Básico do SLURM				###
###########################################################################

#SBATCH --nodes=1                      #Numero de Nós
#SBATCH --ntasks=12                    #Numero total de tarefas MPI/OPENMP
#SBATCH -p dev                         #Fila (partition) a ser utilizada
#SBATCH  -J Pedro_invepsilon                  #Nome job e identificacao

###########################################################################
###				Opcionais do SLURM			###
###########################################################################
#Exibe os nós alocados para o Job
#echo $SLURM_JOB_NODELIST
#nodeset -e $SLURM_JOB_NODELIST
ulimit -c unlimited
ulimit -s unlimited


## Garante que está no diretório certo (local da submissao do job)
cd $SLURM_SUBMIT_DIR

#########################################################################
###			Configura Compiladores			      ###
###			   	   e				      ###
###			      Executaveis			      ###
#########################################################################	

## Descarrega modulos precarredos por padrao
#module unload gnu9/9.4.0
#module unload openmpi4/4.1.1

## Carrega o Necessário
#module load intel/2023.1.0 mpi/2021.9.
module swap intel gnu9/9.4.0
#module load gnu9/9.4.0
#module load openmpi4/4.1.1

module load gcc-11.3.0-gcc-9.4.0-2zeeydc netlib-lapack-3.10.1-gcc-11.3.0-w4ll4ab openblas/0.3.7


#PW_DIR=/home/wendel/packages/Intel/qe-7.3/bin
XATU_DIR=/home/juanjo-uam/work/xatu/bin
#WANNIER_DIR=/home/wendel/packages/Intel/qe-7.2/bin

#########################################################################
###				Jobs MPI				#
#########################################################################

#####
## AJUSTES PARA RODAR NO LOCAL-SCRATCH
#####

	#WORK_DIR=/local-scratch/vinicius/4-AGNR-wa
	

	#mkdir $WORK_DIR
	
	#cp $INPUT $WORK_DIR
	#cd $WORK_DIR

#####
## EXEC DA TAREFA EM SI
#####


#mpiexec -np $SLURM_NTASKS $PW_DIR/pw.x  < scf.in > scf.out
#mpiexec -np $SLURM_NTASKS $PW_DIR/pw.x  < nscf.in > nscf.out
#mpiexec -np $SLURM_NTASKS $PW_DIR/pw.x  < nscf.in > nscf.out
#mpiexec -np 1 $WANNIER_DIR/wannier90.x -pp AGNR
#mpiexec -np $SLURM_NTASKS $WANNIER_DIR/pw2wannier90.x  < AGNR.pw2wan > pw2wan.out
mpiexec -np 1 $XATU_DIR/xatu MoS2.model  MoS2_test.txt -z MoS2_TB_screening.txt > output_MoS2_TB_invepsilon.out
#process 13877


#####
## COPIA DE VOLTA PARA O LOCAL DE SUBMISSÃO
#####

#mv * $SLURM_SUBMIT_DIR


