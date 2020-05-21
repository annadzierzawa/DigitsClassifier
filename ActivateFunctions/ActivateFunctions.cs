using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitsClassifier.ActivateFunctions
{
    class ActivationFunctions
    {
        public static double[] ApplyActFunc(double[] matrix, int p)
        {
            double[] result = new double[p];
            for (int i = 0; i < p; i++)
            {
                result[i] = Tanh(matrix[i]);
            }
            return result;
        }
        public static double[,] ApplyActFunc(double[,] matrix, int p, int m)
        {
            double[,] result = new double[p, m];
            for (int i = 0; i < p; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    result[i, j] = Tanh(matrix[i, j]);
                }
            }
            return result;
        }
        //Hyperbolic tangent
        public static double Tanh(double number)
        {
            return (Math.Pow(Math.E, 2 * number) - 1) / (Math.Pow(Math.E, 2 * number) + 1);
        }
        //Derrivative of the activation function
        public static double TanhDerriv(double number)
        {
            return (1 - Math.Pow(Tanh(number), 2));
        }
    }
}
