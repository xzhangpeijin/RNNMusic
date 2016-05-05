import org.apache.commons.io.FileUtils;
import org.deeplearning4j.examples.recurrent.character.CharacterIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.*;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/

	One minor difference between this example and Karpathy's work:
	The LSTM architectures appear to differ somewhat. GravesLSTM has peephole connections that
	Karpathy's char-rnn implementation appears to lack. See GravesLSTM javadoc for details.
	There are pros and cons to both architectures (addition of peephole connections is a more powerful
	model but has more parameters per unit), though they are not radically different in practice.

	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	from Project Gutenberg. Training on other text sources should be relatively easy to implement.

    For more details on RNNs in DL4J, see the following:
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork
 */
public class MusicLSTM {
    public static List<String> getMusicFiles(String dir) {
        List<String> files = new ArrayList<String>();
        File parent = new File(dir);
        for (File child : parent.listFiles()) {
            if (child.getName().endsWith(".csv")) {
                files.add(child.getAbsolutePath());
            }
        }
        return files;
    }

    public static void trainLSTM() throws IOException {
        int lstmLayerSize = 256;					//Number of units in each GravesLSTM layer
        int miniBatchSize = 100;						//Size of mini batch to use when  training
        int exampleLength = 2000;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 250;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 600;							//Total number of training epochs

        String inputFile = "D:\\GitHub\\RNNMusic\\data\\encoded\\FurElise.csv";
        String output = "FurEliseLoss2.csv";
//        String inputDir = "D:\\GitHub\\RNNMusic\\data\\encoded\\The Classical Collection 3 Chopin - Piano Classics";

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        NoteIterator iter = new NoteIterator(Arrays.asList(inputFile), miniBatchSize, exampleLength);
        int nOut = iter.totalOutcomes();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .learningRate(0.1)
            .dropOut(0.5)
            .rmsDecay(0.95)
            .seed(0)
//            .regularization(true)
//            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
            .list(4)
            .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                .activation("tanh").build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation("tanh").build())
            .layer(2, new DenseLayer.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation("relu").build())
            .layer(3, new RnnOutputLayer.Builder(LossFunction.MSE).activation("identity")
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for( int i=0; i<layers.length; i++ ){
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        PrintWriter out = new PrintWriter(new FileWriter(new File(output)));

        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        out.println("Batch,Loss");
        for( int i=0; i<numEpochs; i++ ){
            System.out.println("Epoch " + i);

            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                out.format("%d,%f%n", i, net.score());
                out.flush();
            }

            iter.reset();	//Reset iterator for another epoch

            if ((i + 1) % 50 == 0 && !Double.isNaN(net.score())) {
                //Write the network parameters:
                try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get("coefficients.bin")))){
                    Nd4j.write(net.params(),dos);
                }
                //Write the network configuration:
                FileUtils.write(new File("conf.json"), net.getLayerWiseConfigurations().toJson());
            }
        }

        System.out.println("\n\nTraining complete");

        System.out.println("--------------------");
    }

    public static void synthMusic() throws IOException {
        String inputFile = "D:\\GitHub\\RNNMusic\\data\\encoded\\FurElise.csv";
        String outputFile = "SynthMusic_FE3.csv";

        int seedLength = 500;
        double time = 0.02;
        double timegap = 0.01;

        NoteIterator iter = new NoteIterator(Arrays.asList(inputFile), 1, seedLength);

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));

        //Load parameters from disk:
        INDArray newParams;
        try(DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"))){
            newParams = Nd4j.read(dis);
        }

        //Create a MultiLayerNetwork from the saved configuration and parameters
        MultiLayerNetwork net = new MultiLayerNetwork(confFromJson);
        net.init();
        net.setParameters(newParams);

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(1, iter.inputColumns(), seedLength);
        for (int i = 0; i < seedLength; i++) {
            for (int j = 0; j < iter.inputColumns(); j++) {
                initializationInput.putScalar(new int[]{0, j, i}, iter.data.get(i)[j]);
            }
        }

        PrintWriter out = new PrintWriter(new FileWriter(new File(outputFile)));
        out.println(iter.header);
        for (int i = 0; i < seedLength; i++) {
            out.print(time);
            for (int j = 0; j < iter.inputColumns(); j++) {
                out.print("," + iter.data.get(i)[j]);
            }
            out.println();
            time += timegap;
        }
        out.flush();

        System.out.println(initializationInput.size(0));
        net.rnnClearPreviousState();
        net.setInput(initializationInput);
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

        for (int i = seedLength; i < iter.data.size(); i++) {
            INDArray nextInput = Nd4j.zeros(1, iter.inputColumns());
            double[] data = new double[iter.totalOutcomes()];

            out.print(time);
            for (int j = 0; j < iter.totalOutcomes(); j++) {
                data[j] = output.getDouble(0, j);
                nextInput.putScalar(new int[]{0, j}, data[j]);
                out.print("," + data[j]);
            }
            out.println();
            out.flush();

            output = net.rnnTimeStep(nextInput);
            time += timegap;
        }

        System.out.println("done synthesis");
    }

	public static void main( String[] args ) throws Exception {
        trainLSTM();
        synthMusic();
    }
}
