using DigitsClassifier.Normalization;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitsClassifier
{

    public static class Reader
    {
        public static bool Testing = false;
        public static bool LabelReaderRunning = false;
        public static bool ImageReaderRunning = false;

        private static string TrainImagePath = @"D:\Semestr 4\Systemy sztucznej inteligencji\REPOZYTORIUM\Digits Classifier\database\train-images.idx3-ubyte";         //Each feature vector (row in the feature matrix) consists of 784 pixels (intensities) -- unrolled from the original 28x28 pixels images.
        private static string TrainLabelPath = @"D:\Semestr 4\Systemy sztucznej inteligencji\REPOZYTORIUM\Digits Classifier\database\train-labels.idx1-ubyte";         //Uniformly distributed class labels 0-9 corresponding to the respective handwritten digit shown in the image.
        /*the same for testing data*/
        private static string TestLabelPath = @"D:\Semestr 4\Systemy sztucznej inteligencji\REPOZYTORIUM\Digits Classifier\database\t10k-labels.idx1-ubyte";
        private static string TestImagePath = @"D:\Semestr 4\Systemy sztucznej inteligencji\REPOZYTORIUM\Digits Classifier\database\t10k-images.idx3-ubyte";

        private static string LabelPath = Testing ? TestLabelPath : TrainLabelPath;
        private static string ImagePath = Testing ? TestImagePath : TrainImagePath;
        static int LabelOffset = 8;                            //Label data start from 8 byte
        static int ImageOffset = 16;                        //Image data start from 16 byte
        static int Resolution = 28;


        public static int ReadNextLabel()
        {
            if (LabelReaderRunning) { throw new Exception("Already accessing file"); }

            FileStream fs = File.OpenRead(LabelPath);
            
            if (LabelOffset > fs.Length) { 
                LabelOffset = 8; 
                ImageOffset = 16;                       //starts reading training data from the beginning
            }

            fs.Position = LabelOffset;                   //setting file stream reading position to the current labels offset

            int result = -1;
            try
            {
                result = fs.ReadByte();         //reading the category of image(it is saved on one byte)
            }
            catch (Exception ex) { 
                Console.WriteLine(ex.ToString()); 
            }
           
            LabelOffset++;              //incerement labels offset to the address of next label
            fs.Close();

            return result;
        }

        public static double[,] ReadNextImage()
        {
            if (ImageReaderRunning) { throw new Exception("Already accessing file"); }

            FileStream fs = File.OpenRead(ImagePath);           //Read image

            if (ImageOffset > fs.Length) { 
                ImageOffset = 16; 
                LabelOffset = 8;                        //starts reading training data from the beginning
            }

            fs.Position = ImageOffset;

            byte[] image = new byte[Resolution * Resolution];       //28*28 = 784

            try
            {
                fs.Read(image, 0, Resolution * Resolution);
            }
            catch (Exception ex) { 
                Console.WriteLine(ex.ToString()); 
            }

            int[] array = Array.ConvertAll(image, Convert.ToInt32);

            ImageOffset += Resolution * Resolution;                      //adding 784 to offset of previous image 
           
            double[,] image2d = new double[Resolution, Resolution];     //two dimensional array that stores the pixels of the image
            
            for (int i = 0; i < Resolution; i++)
            {
                for (int j = 0; j < Resolution; j++)
                {
                    image2d[i, j] = (double)array[(Resolution * i) + j];                //Fulfilling of two dimensional array of doubles
                }
            }
            
            Normalizer.Normalize(image2d, true, Resolution, Resolution);        //Normalize the result matrix

            fs.Close();
            return image2d;
        }

    }
}
