using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitsClassifier
{
    class Program
    {
        /*public static bool trainMode = true;*/ // true training, false testing
        public static bool trainMode = false;
        static void Main(string[] args)
        {
            //ProgramManager.reset();                     
            NetworkParametersHandler.ReadWeightBias();             //Reading network parameters form file
            if (trainMode)
            {
                while (!ProgramManager.finished)
                {
                    ProgramManager.Training();
                }
            }
            else { ProgramManager.Testing(); }

           
            Console.WriteLine("Finished");
            Console.ReadKey();
        }
    }
}
