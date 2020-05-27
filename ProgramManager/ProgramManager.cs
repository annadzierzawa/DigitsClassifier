using DigitsClassifier.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitsClassifier
{
    public class ProgramManager
    {
        public static bool isrunning = false;
        public static bool finished = false;
        static double SaveState = 100;
        static double avg = 0;
        //static double maxavg = 0;
        static double avgerror = 0;
        static int batchsize = 5;
        static double iterator = 0;
        //Reset (re-initialize) weights and biases of the neural network
        public static void reset()
        {
            Network nn = new Network();
            nn.initialize();
            NetworkParametersHandler.WriteWeightBias();
        }
        public static void Testing()
        {
            Network nn = new Network();
            while (iterator < 9000)
            {
                iterator++;
                nn.Calculate(Reader.ReadNextImage());
                int correct = Reader.ReadNextLabel();
                double certainty = -99.0;
                int guess = -1;

                for (int i = 0; i < 10; i++) 
                { 
                    if (nn.OutputValues[i] > certainty)
                    { 
                        certainty = nn.OutputValues[i];
                        guess = i; 
                    }
                }
                avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0.0);
                Console.WriteLine("Guess: " + guess + " Correct: " + correct + " Correct? " + (guess == correct ? "Yes " : "No ").PadRight(4) + " %Correct: " + Math.Round(avg, 10).ToString().PadRight(12));
            }
        }
        public static void Training()
        {
            for (int i = 0; i < SaveState; i++)
            {
                List<Network> ntw = new List<Network>();
                var tasks = new Task[batchsize];
                for (int j = 0; j < batchsize; j++)
                {
                    //B/c ii may change and this code can't let that happen
                    int iterator = j;
                    double[,] image = Reader.ReadNextImage();
                    int correct = Reader.ReadNextLabel();
                    ntw.Add(new Network());
                    tasks[iterator] = Task.Run(() => ntw[iterator].Backprop(image, correct));
                }
                Task.WaitAll(tasks);
                //Syncronously descend
                foreach (Network nn in ntw)
                {
                    nn.Descent();
                }
                //Updating the weights with the avg gradients
                Network.Descent(batchsize);
                UserValidation();
            }
            //Save weights and biases
            NetworkParametersHandler.WriteWeightBias();
        }
        public static void UserValidation()
        {
            Network nn = new Network(); double[,] image; int correct;

            //Some user validation code

            image = Reader.ReadNextImage();
            correct = Reader.ReadNextLabel();
            //Backprop again for averaging
            nn.Calculate(image);


            int guess = 0; 
            double certainty = 0;
           
            for (int i = 0; i < 10; i++) 
            {
                if (nn.OutputValues[i] > certainty) //finding the biggest value in output layer
                { 
                    certainty = nn.OutputValues[i]; 
                    guess = i; 
                }
            }
            //Calculate the moving average of the percentage of trials correct of those written to console
            double error = 0;
            for (int i = 0; i < Network.OutputCount; i++)
            {
                error += ((i == correct ? 1.0 : 0.0) - nn.OutputValues[i]) * ((i == correct ? 1.0 : 0.0) - nn.OutputValues[i]);
            }
            iterator++;
            avgerror = ((iterator / (iterator + 1)) * avgerror) + ((1 / iterator) * error);
            avg = (avg * (iterator / (iterator + 1))) + ((guess == correct) ? (1 / iterator) : 0.0);

            //Print various things to the console for verification that things are nominal
            Console.WriteLine("Correct: " + correct + "\t" + " Guess: " + guess + "\t" + " Correct? " + (guess == correct ? "Yes " : "No ").ToString().PadRight(3) + "\t"  + " %Correct: " + "\t" + Math.Round(avg, 5).ToString().PadRight(7));

            //Reset the console data every few iterations to ensure up to date numbers
            if (iterator > 1000)
            { 
                iterator = 100; 
                Network.Epoch++; 
            }
        }
    }
}
