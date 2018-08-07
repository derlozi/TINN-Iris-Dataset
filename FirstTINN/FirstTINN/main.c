#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "tinn-master/Tinn.h"
#include <mem.h>

#define LINESIZE 40

struct dataset
{
    struct data* set;
    int elementcount;
};

struct data// contains one set of features -> the properties of one flower
{
    float slen;
    float swid;
    float plen;
    float pwid;
    float isSetosa;
    float isVersicolor;
    float isVirginica;
};

float rate = 0.3; //learning rate

struct dataset addDataset(struct dataset data, int elementnr, char* line)// adds the properties of one flower to the set
{
    char* delimiter = ",";
    char* sp;

    data.set = (struct data*)realloc(data.set, (elementnr+1)*sizeof(*(data.set)));
    (data.set+elementnr)->slen = (float)atof(strtok(line, delimiter));
    (data.set+elementnr)->swid = (float)atof(strtok(NULL, delimiter));
    (data.set+elementnr)->plen = (float)atof(strtok(NULL, delimiter));
    (data.set+elementnr)->pwid = (float)atof(strtok(NULL, delimiter));
    printf("%f,%f,%f,%f,", (data.set+elementnr)->slen, (data.set+elementnr)->swid, (data.set+elementnr)->plen, (data.set+elementnr)->pwid);

    sp = strtok(NULL, delimiter);
    printf("%s\n", sp);
    if(strcmp(sp, "Iris-setosa\n") == 0)
    {
        (data.set+elementnr)->isSetosa = 1;
        (data.set+elementnr)->isVirginica = 0;
        (data.set+elementnr)->isVersicolor = 0;
    }
    if(strcmp(sp, "Iris-versicolor\n") == 0)
    {
        (data.set+elementnr)->isSetosa = 0;
        (data.set+elementnr)->isVirginica = 0;
        (data.set+elementnr)->isVersicolor = 1;
    }
    if(strcmp(sp, "Iris-virginica\n") == 0)
    {
        (data.set+elementnr)->isSetosa = 0;
        (data.set+elementnr)->isVirginica = 1;
        (data.set+elementnr)->isVersicolor = 0;
    }
    return data;
}

struct dataset initializeData(char* filename)//reads a textfile with the dataset into the program and stores it
{
    struct dataset tdata;
    tdata.set = malloc(1);
    char* line = calloc(LINESIZE, sizeof(char));
    FILE *fp;
    int linecount = 0;

    fp = fopen(filename, "r");
    if(fp==NULL)
    {
        printf("Opening the data file failed\n");
    }
    printf("Dumping the data:\n");
    while(fgets(line, LINESIZE, fp))
    {
        printf("%s", line);
        tdata = addDataset(tdata, linecount, line);
        linecount++;
    }
    tdata.elementcount = linecount;
    fclose(fp);
    printf("The file had %i lines\n", linecount);
    return tdata;
}


float* train(Tinn* netp, struct data* tdata, int datacount, float* errors)
{

    for(int c  = 0; c< datacount; c++)
    {
        float in[4] = {(tdata + c)->slen, (tdata + c)->swid, (tdata + c)->plen, (tdata + c)->pwid};
        float out[3] = {(tdata + c)->isSetosa, (tdata + c)->isVersicolor, (tdata + c)->isVirginica};
        errors[c] = xttrain(*netp, in, out, rate);
    }
    return errors;
}


float getError(float* errors, int datacount)//averages all errors over one trainig session
{
    float sum=0;
    for(int c = 0; c<datacount; c++)
    {

        sum =sum+ errors[c];
    }

    return sum/datacount;
}

int main() {

    float toterror = 1;//initialize it somewhere over the smallest alloweded error to start training
    int it = 0;

    Tinn net =  xtbuild(4,10,3);
    Tinn* netp = &net;

    struct dataset traindata = initializeData("C:\\Users\\loren\\Documents\\Programmieren\\C\\FirstTINN\\traindata.txt");
    float* errors = (float*)calloc(traindata.elementcount, sizeof(float));
    printf("Press a key to start learning");
    getchar();
    while(toterror>0.0015)
    {
        errors = train(netp, traindata.set, traindata.elementcount, errors);
        toterror = getError(errors, traindata.elementcount);
        printf("Training Iteration NR:%i, error is:%f\n", it, toterror);
        it++;
    }
    printf("The net was successfully trained, the last error was:%f", toterror);
    return 0;
}

