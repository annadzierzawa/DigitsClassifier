using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitsClassifier.Normalization
{
    class Normalizer
    {
        public static double[] Normalize(double[] array, bool image)
        {
            if (!image)
            {
                double max = 0;
                double min = 0;
                double a = 0; 
                double b = 0.0001;

                foreach (double item in array) 
                {
                    if (item > max) 
                    {
                        max = item; 
                    }
                    if (item < min) 
                    { 
                        min = item; 
                    } 
                }
                for (int i = 0; i < array.Length; i++)
                {
                    array[i] = (array[i] < 0 ? -1 : 1) * (a + ((array[i] - min) * (b - a) / (max - min)));                   //Data normalization
                }
            }
            else
            {
                double mean = 0;                               
                double stdDev = 0;                            //Standard deviation

                //Calc average of data
                foreach (double item in array) 
                {
                    mean += item; 
                }         
                mean /= array.Length;

                //Calc standard deviation of data
                foreach (double item in array)
                {                                                      
                    stdDev += (item - mean) * (item - mean);            
                }
                stdDev /= array.Length;
                stdDev = Math.Sqrt(stdDev);


                if (stdDev == 0) 
                {
                    stdDev = 0.000001;          //Prevent divide by zero b/c of sigma = 0
                }           
              

                //Calc zscore
                for (int i = 0; i < array.Length; i++)
                {
                    array[i] = (array[i] - mean) / stdDev;
                }
            }
            return array;
        }
        
        public static double[,] Normalize(double[,] array, bool image, int depth, int count)
        {
            double[] smallArray = new double[depth * count];

            int iterator = 0;

            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < count; j++)
                {
                    smallArray[iterator] = array[i, j];
                    iterator++;
                }
            }

            smallArray = Normalize(smallArray, image);

            iterator = 0;

            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < count; j++)
                {
                    array[i, j] = smallArray[iterator];
                    iterator++;
                }
            }
            return array;
        }
        
        public static double[,,] Normalize(double[,,] array, bool image, int depth, int count1, int count2)
        {
            double[] workingvalues = new double[depth * count1 * count2];

            int iterator = 0;

            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < count1; j++)
                {
                    for (int k = 0; k < count2; k++)
                    {
                        workingvalues[iterator] = array[i, j, k];
                        iterator++;
                    }
                }
            }

            workingvalues = Normalize(workingvalues, image);

            iterator = 0;

            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < count1; j++)
                {
                    for (int k = 0; k < count2; k++)
                    {
                        array[i, j, k] = workingvalues[iterator];
                        iterator++;
                    }
                }
            }
            return array;
        }
    }
}
