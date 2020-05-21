using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DigitsClassifier.NeuralNetwork;

namespace DigitsClassifier
{
    public class NetworkParametersHandler
    {

        const string Path = @"D:\Semestr 4\Systemy sztucznej inteligencji\REPOZYTORIUM\Digits Classifier\spagghetti\DigitRecog-master\DataBackup.txt";
        static bool Running = false;
        
        /*Reading parameters of network to file*/
        public static void ReadWeightBias()
        {
            if (Running == true) { throw new Exception("Already accessing file"); }

            Running = true;

            FileStream fs = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.None);   
            StreamReader sr = new StreamReader(fs);
            string all = sr.ReadToEnd();             //Read all file
            string[] weightStrings = all.Split(' ');

            int iterator = 0;

            //Read input weights
            for (int i = 0; i < Network.InputCount; i++)
            {
                for (int j = 0; j < Network.Resolution * Network.Resolution; j++)
                {
                    double.TryParse(weightStrings[iterator], out double weight);
                    Network.InputWeights[i, j] = weight;
                    iterator++;
                }
            }

            //Read hidden weights
            for (int i = 0; i < Network.HiddenDepth; i++)
            {
                for (int j = 0; j < Network.HiddenCount; j++)
                {
                    if (i == 0)
                    {
                        for (int k = 0; k < Network.InputCount; k++)
                        {
                            double.TryParse(weightStrings[iterator], out double weight);
                            Network.FirstHiddenWeights[j,k] = weight;
                            iterator++;
                        }
                    }
                    else
                    {
                        for (int k = 0; k < Network.HiddenCount; k++)
                        {
                            double.TryParse(weightStrings[iterator], out double weight);
                            Network.HiddenWeights[i - 1, j, k] = weight;
                            iterator++;
                        }
                    }
                }
            }

            //Read output weights
            for (int i = 0; i < Network.OutputCount; i++)
            {
                for (int j = 0; j < Network.HiddenCount; j++)
                {
                    double.TryParse(weightStrings[iterator], out double weight);
                    Network.OutputWeights[i, j] = weight;
                    iterator++;
                }
            }

            //Read input biases
            for (int i = 0; i < Network.InputCount; i++)
            {
                double.TryParse(weightStrings[iterator], out double bias);
                Network.InputBiases[i] = bias;
                iterator++;
            }

            //Read hidden biases
            for (int i = 0; i < Network.HiddenDepth; i++)
            {
                for (int j = 0; j < Network.HiddenCount; j++)
                {
                    double.TryParse(weightStrings[iterator], out double bias);
                    Network.HiddenBiases[i, j] = bias;
                    iterator++;
                }
            }
            sr.Close();
            fs.Close();
            Running = false;
        }

        /*Writing parameters of network to file*/
        public static void WriteWeightBias()
        {
            if (Running == true) { throw new Exception("Already accessing file"); }

            Running = true;

            FileStream fs = new FileStream(Path, FileMode.Create, FileAccess.Write, FileShare.None);
            StreamWriter sw = new StreamWriter(fs);

            //Write input weights
            for (int i = 0; i < Network.InputCount; i++)
            {
                for (int j = 0; j < Network.Resolution * Network.Resolution; j++)
                {
                    sw.Write(Network.InputWeights[i, j].ToString() + " ");
                }
            }

            //Write hidden weights
            for (int i = 0; i < Network.HiddenDepth; i++)
            {
                for (int j = 0; j < Network.HiddenCount; j++)
                {
                    if (i == 0)
                    {
                        for (int k = 0; k< Network.InputCount; k++)
                        {
                            sw.Write(Network.FirstHiddenWeights[j, k].ToString() + " ");
                        }
                    }
                    else
                    {
                        for (int k = 0; k < Network.HiddenCount; k++)
                        {
                            sw.Write(Network.HiddenWeights[i - 1, j, k].ToString() + " ");
                        }
                    }
                }
            }

            //Write output weights
            for (int i = 0; i < Network.OutputCount; i++)
            {
                for (int j = 0; j < Network.HiddenCount; j++)
                {
                    sw.Write(Network.OutputWeights[i, j].ToString() + " ");
                }
            }

            //Write input biases
            for (int i = 0; i < Network.InputCount; i++)
            {
                sw.Write(Network.InputBiases[i].ToString() + " ");
            }

            //Write hidden biases
            for (int i = 0; i < Network.HiddenDepth; i++)
            {
                for (int j = 0; j < Network.HiddenCount; j++)
                {
                    sw.Write(Network.HiddenBiases[i, j].ToString() + " ");
                }
            }
            sw.Close();
            fs.Close();
            Running = false;
        }

    }
}
