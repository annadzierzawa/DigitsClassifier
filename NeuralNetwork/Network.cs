using DigitsClassifier.ActivateFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitsClassifier.NeuralNetwork
{
    public class Network /*: IDisposable*/
    {
        public static int Epoch = 0;                //Number of iteration

        public static int HiddenDepth = 2;        //Numberof the hidden layers
        
        public static int Depth = HiddenDepth + 2;          //Hiddendepth + input [1] + output [1]- Number of all layers  
        public static int HiddenCount = 19;                  //Count of neurons per layer
        public static int OutputCount = 10;                 //Count of output neurons
        public static int Resolution = 28;                  //Number of pixels in one 'row'. Our pitures are 28X28 - PK
        public static int InputCount = Resolution;
        
        
        public static double AvgGradient = 0;
        
        private static double LearningRateDecay = (0.000146 / 5.0) * (double)Epoch;
        private static double LearningRate = 0.0000146 - LearningRateDecay;
        public static double Momentum = 0.9;
        //Overall gradients
        static double[,] AvgInputWeightGradient = new double[InputCount, Resolution * Resolution];
        static double[,] AvgFirstHiddenWeightGradient = new double[HiddenCount, InputCount];
        static double[,,] AvgHiddenWeightGradient = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
        static double[,] AvgOutputWeightGradient = new double[OutputCount, HiddenCount];
        static double[,] AvgHiddenBiasGradient = new double[HiddenDepth, HiddenCount];
        static double[] AvgInputBiasGradient = new double[InputCount];
        
        double[] InputErrorSignals { get; set; }
        double[,] HiddenErrorSignals { get; set; }
        double[] OutputErrorSignals { get; set; }
     
        static public double[,] InputWeights = new double[InputCount, Resolution * Resolution];
        static public double[,] FirstHiddenWeights = new double[HiddenCount, InputCount];
        static public double[,,] HiddenWeights = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
        static public double[,] OutputWeights = new double[OutputCount, HiddenCount];
        
        public static double[] InputBiases = new double[InputCount];
        public static double[,] HiddenBiases = new double[HiddenDepth, HiddenCount];
       
        double[] InputZVals { get; set; }
        double[,] HiddenZVals { get; set; }
        double[] OutputZVals { get; set; }
        
        double[,] InputWeightGradient { get; set; }
        double[,] FirstHiddenWeightGradient { get; set; }
        double[,,] HiddenWeightGradient { get; set; }
        double[,] OutputWeightGradient { get; set; }
       
        static double[,] InputWeightMomentum = new double[InputCount, Resolution * Resolution];
        static double[,] FirstHiddenWeightMomentum = new double[HiddenCount, InputCount];
        static double[,,] HiddenWeightMomentum = new double[HiddenDepth, HiddenCount, HiddenCount];
        static double[,] OutputWeightMomentum = new double[OutputCount, HiddenCount];
        static double[] InputBiasMomentum = new double[InputCount];
        static double[,] HiddenBiasMomentum = new double[HiddenDepth, HiddenCount];
        
        double[] InputValues { get; set; }
        double[,] HiddenValues { get; set; }
       
        public double[] OutputValues = new double[OutputCount];
        
        public static void Descent(int batchsize)
        {
            
            AvgGradient = 0;                //Reset avg gradient

            //Input
            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < Resolution * Resolution; j++)
                {
                    InputWeights[i, j] -= LearningRate * AvgInputWeightGradient[i, j] * (-2 / (double)batchsize);
                    AvgGradient -= LearningRate * AvgInputWeightGradient[i, j] * (-2 / (double)batchsize);
                }
                InputBiases[i] -= LearningRate * AvgInputBiasGradient[i] * (-2 / (double)batchsize);
            }

            //Hidden
            for (int i = 0; i < HiddenDepth; i++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    if (i == 0)
                    {
                        for (int k = 0; k < InputCount; k++)
                        {
                            FirstHiddenWeights[j, k] -= LearningRate * AvgFirstHiddenWeightGradient[j, k] * (-2 / (double)batchsize);
                            AvgGradient -= LearningRate * AvgFirstHiddenWeightGradient[j, k] * (-2 / (double)batchsize);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < HiddenCount; k++)
                        {
                            HiddenWeights[i - 1, j, k] -= LearningRate * AvgHiddenWeightGradient[i - 1, j, k] * (-2 / (double)batchsize);
                            AvgGradient -= LearningRate * AvgHiddenWeightGradient[i - 1, j, k] * (-2 / (double)batchsize);
                        }
                    }
                    HiddenBiases[i, j] -= LearningRate * AvgHiddenBiasGradient[i, j] * (-2 / (double)batchsize);
                }
            }
            //Output
            for (int i = 0; i < OutputCount; i++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    OutputWeights[i, j] -= LearningRate * AvgOutputWeightGradient[i, j] * (-2 / (double)batchsize);
                    AvgGradient -= LearningRate * AvgOutputWeightGradient[i, j] * (-2 / (double)batchsize);
                }
            }
            AvgGradient /= AvgHiddenWeightGradient.Length + AvgInputWeightGradient.Length + AvgOutputWeightGradient.Length;

            //Reset averages
            AvgInputWeightGradient = new double[InputCount, Resolution * Resolution];
            AvgFirstHiddenWeightGradient = new double[HiddenCount, InputCount];
            AvgHiddenWeightGradient = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
            AvgOutputWeightGradient = new double[OutputCount, HiddenCount];
            AvgHiddenBiasGradient = new double[HiddenDepth, HiddenCount];
            AvgInputBiasGradient = new double[InputCount];
        }
        //Stochastic descent (all code below is done according to formulas)
        //This adds each NN's gradients to the avg
        public void Descent()
        {
            //Input
            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < Resolution * Resolution; j++)
                {
                    //Nesterov momentum
                    InputWeightMomentum[i, j] = (InputWeightMomentum[i, j] * Momentum) - (LearningRate * InputWeightGradient[i, j]);
                    AvgInputWeightGradient[i, j] += InputWeightGradient[i, j] + InputWeightMomentum[i, j];
                }
                double tempbias = InputErrorSignals[i] * ActivationFunctions.TanhDerriv(InputZVals[i]);
                InputBiasMomentum[i] = (InputBiasMomentum[i] * Momentum) - (LearningRate * tempbias);
                AvgInputBiasGradient[i] += tempbias + InputBiasMomentum[i];
            }
            //Hidden
            for (int i = 0; i < HiddenDepth; i++)
            {
                for (int j = 0; j < HiddenCount;j++)
                {
                    if (i == 0)
                    {
                        for (int k = 0; k < InputCount; k++)
                        {
                            //Nesterov momentum
                            FirstHiddenWeightMomentum[j, k] = (FirstHiddenWeightMomentum[j, k] * Momentum) - (LearningRate * FirstHiddenWeightGradient[j, k]);
                            AvgFirstHiddenWeightGradient[j, k] += FirstHiddenWeightGradient[j, k] + FirstHiddenWeightMomentum[j, k];
                        }
                    }
                    else
                    {
                        for (int k = 0; k < HiddenCount; k++)
                        {
                            //Nesterov momentum
                            HiddenWeightMomentum[i - 1, j, k] = (HiddenWeightMomentum[i - 1, j, k] * Momentum) - (LearningRate * HiddenWeightGradient[i - 1, j, k]);
                            AvgHiddenWeightGradient[i - 1, j, k] += HiddenWeightGradient[i - 1, j, k] + HiddenWeightMomentum[i - 1, j, k];
                        }
                    }
                    double tempbias = HiddenErrorSignals[i, j] * ActivationFunctions.TanhDerriv(HiddenZVals[i, j]);
                    HiddenBiasMomentum[i, j] = (HiddenBiasMomentum[i, j] * Momentum) - (LearningRate * tempbias);
                    AvgHiddenBiasGradient[i, j] += tempbias + HiddenBiasMomentum[i, j];
                }
            }
            //Output
            for (int i = 0; i < OutputCount; i++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    //Nesterov momentum
                    OutputWeightMomentum[i, j] = (OutputWeightMomentum[i, j] * Momentum) - (LearningRate * OutputWeightGradient[i, j]);
                    AvgOutputWeightGradient[i, j] += OutputWeightGradient[i, j] + OutputWeightMomentum[i, j];
                }
            }
        }
        
        public void Backprop(double[,] image, int correct)
        {
            //Forward propagation of data
            Calculate(image);

            //Reset things about to be calculated
            InputErrorSignals = new double[InputCount];
            HiddenErrorSignals = new double[HiddenDepth, HiddenCount];
            OutputErrorSignals = new double[OutputCount];
            InputWeightGradient = new double[InputCount, Resolution * Resolution];
            FirstHiddenWeightGradient = new double[HiddenCount, InputCount];
            HiddenWeightGradient = new double[HiddenDepth - 1, HiddenCount, HiddenCount];
            OutputWeightGradient = new double[OutputCount, HiddenCount];

            //Output
            //Foreach ending neuron
            for (int k = 0; k < OutputCount; k++)
            {
                double upperlayerderiv = 2.0 * ((k == correct ? 1.0 : 0.0) - OutputValues[k]);
                OutputErrorSignals[k] = upperlayerderiv;

                //Calculate gradient
                //This works b/c of only 1 hidden layer, will need to be changed if HiddenDepth is modified
                for (int j = 0; j < HiddenCount; j++)
                {
                    OutputWeightGradient[k, j] = HiddenValues[HiddenDepth - 1, j] * ActivationFunctions.TanhDerriv(OutputZVals[k]) * OutputErrorSignals[k];
                }
            }
            //Hidden
            //Foreach layer of hidden 'neurons'
            //Calc errors
            for (int l = HiddenDepth - 1; l >= 0; l--)
            {
                //Hidden upper layer derrivative calculation
                //Foreach starting neuron
                if (l == HiddenDepth - 1)
                {
                    for (int k = 0; k < HiddenCount; k++)
                    {
                        double upperlayerderiv = 0;

                        //Foreach ending neuron
                        for (int j = 0; j < OutputCount; j++)
                        {
                            //Hiddenweights uses l because the formula's l + 1 is l due to a lack of input layer in this array
                            upperlayerderiv += OutputWeights[j, k] * ActivationFunctions.TanhDerriv(OutputZVals[j]) * OutputErrorSignals[j];
                        }
                        HiddenErrorSignals[l, k] = upperlayerderiv;
                    }
                }
                else
                {
                    for (int k = 0; k < HiddenCount; k++)
                    {
                        double upperlayerderiv = 0;
                        //Foreach ending neuron
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            //Hiddenweights uses l instead of l + 1 because firsthiddenweights is a different array
                            upperlayerderiv += HiddenWeights[l, j, k] * ActivationFunctions.TanhDerriv(HiddenZVals[l + 1, j]) * HiddenErrorSignals[l + 1, j];
                        }
                        HiddenErrorSignals[l, k] = upperlayerderiv;
                    }
                }
            }
            //Calc values
            for (int l = 0; l < HiddenDepth; l++)
            {
                //Foreach starting neuron
                for (int k = 0; k < HiddenCount; k++)
                {
                    if (l == 0)
                    {
                        //Foreach ending neuron neuron
                        for (int j = 0; j < InputCount; j++)
                        {
                            FirstHiddenWeightGradient[k, j] = InputValues[j] * ActivationFunctions.TanhDerriv(HiddenZVals[l, k]) * HiddenErrorSignals[l, k];
                        }
                    }
                    else
                    {
                        //Foreach ending neuron neuron
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            HiddenWeightGradient[l - 1, k, j] = HiddenValues[l - 1, j] * ActivationFunctions.TanhDerriv(HiddenZVals[l, k]) * HiddenErrorSignals[l, k];
                        }
                    }
                }
            }
            //Input
            //Foreach starting neuron
            for (int k = 0; k < InputCount; k++)
            {
                double upperlayerderiv = 0;

                //Calculate error signal
                //Foreach ending neuron
                for (int j = 0; j < HiddenCount; j++)
                {
                    upperlayerderiv += FirstHiddenWeights[j, k] * ActivationFunctions.TanhDerriv(HiddenZVals[0, j]) * HiddenErrorSignals[0, j];
                }
                InputErrorSignals[k] = upperlayerderiv;

                //Calculate gradient
                for (int j = 0; j < Resolution * Resolution; j++)
                {
                    InputWeightGradient[k, j] = image[j / Resolution, j - ((j / Resolution) * Resolution)] * ActivationFunctions.TanhDerriv(InputZVals[k]) * InputErrorSignals[k];
                }
            }
        }

        public void Calculate(double[,] image)
        {
            //Reset ZVals (raw values untouched by the activation function), vals, and momentums
            InputZVals = new double[InputCount]; 
            InputValues = new double[InputCount];
            HiddenZVals = new double[HiddenDepth, HiddenCount];
            HiddenValues = new double[HiddenDepth, HiddenCount];
            OutputZVals = new double[OutputCount]; 
            OutputValues = new double[OutputCount];

            //Random r = new Random();
            //Random is used for dropout of neurons, but said feature is currently disabled for efficiency reasons

            //Input
            for (int k = 0; k < InputCount; k++)
            {
                for (int j = 0; j < (Resolution * Resolution); j++)
                {
                    InputZVals[k] += ((InputWeights[k, j] + InputWeightMomentum[k, j]) * image[j / Resolution, j - ((j / Resolution) * Resolution)]) + InputBiases[k];
                }
                InputValues[k] = ActivationFunctions.Tanh(InputZVals[k]);
            }
            //Hidden
            for (int l = 0; l < HiddenDepth; l++)
            {
                for (int k = 0; k < HiddenCount; k++)
                {
                    if (l == 0)
                    {
                        for (int j = 0; j < InputCount; j++)
                        {
                            HiddenZVals[l, k] += (((FirstHiddenWeights[k, j] + FirstHiddenWeightMomentum[k, j]) * InputValues[j]) + HiddenBiases[l, k]);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            //Hiddenweights and momentum use l - 1 because the first layer is under firsthidden and firstmomentum respectively
                            HiddenZVals[l, k] += (((HiddenWeights[l - 1, k, j] + HiddenWeightMomentum[l - 1, k, j]) * HiddenValues[l - 1, j]) + HiddenBiases[l, k]);
                        }
                    }
                    HiddenValues[l, k] = ActivationFunctions.Tanh(HiddenZVals[l, k]);
                }
            }
            //Output
            for (int k = 0; k < OutputCount; k++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    OutputZVals[k] += ((OutputWeights[k, j] + OutputWeightMomentum[k, j]) * HiddenValues[HiddenDepth - 1, j]);
                }
                //No activation function on outputs
                OutputValues[k] = OutputZVals[k];
            }
        }

        public void initialize()
        {
            Random r = new Random();
            //Input
            for (int i = 0; i < InputCount; i++)
            {
                for (int j = 0; j < (Resolution * Resolution); j++)
                {
                    //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                    InputWeights[i, j] = (r.NextDouble() > 0.5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3.0 / (double)(Resolution * Resolution));
                }
            }
            //Hidden
            for (int l = 0; l < HiddenDepth; l++)
            {
                for (int i = 0; i < HiddenCount; i++)
                {
                    if (l == 0)
                    {
                        for (int j = 0; j < InputCount; j++)
                        {
                            //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                            FirstHiddenWeights[i, j] = (r.NextDouble() > 0.5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3.0 / (InputCount * InputCount));
                        }
                    }
                    else
                    {
                        for (int j = 0; j < HiddenCount; j++)
                        {
                            //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                            HiddenWeights[l - 1, i, j] = (r.NextDouble() > 0.5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3.0 / (HiddenCount * HiddenCount));
                        }
                    }
                }
            }
            //Output
            for (int i = 0; i < OutputCount; i++)
            {
                for (int j = 0; j < HiddenCount; j++)
                {
                    //Lecun initialization (draw a rand num from neg limit to pos limit where lim is the sqrt term)
                    OutputWeights[i, j] = (r.NextDouble() > 0.5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3.0 / (double)(HiddenCount * HiddenCount));
                }
            }
        }
    }
}
