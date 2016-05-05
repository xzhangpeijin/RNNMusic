import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;

/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class NoteIterator implements DataSetIterator {
    private int exampleLength;
    private int miniBatchSize;

    private int fileNum = 0;
    private List<String> files;
    public List<double[]> data = new ArrayList<double[]>();
    public String header;

    private int features;

    private LinkedList<Integer> startOffsets = new LinkedList<Integer>();

    public NoteIterator(List<String> files, int miniBatchSize, int exampleLength) throws IOException {
        if (files.size() <= 0) {
            throw new IllegalArgumentException("Must have at least one file");
        }

        for (String textFilePath : files) {
            if (!new File(textFilePath).exists())
                throw new IOException("Could not access file (does not exist): " + textFilePath);
        }
        if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");

		this.exampleLength = exampleLength;
		this.miniBatchSize = miniBatchSize;
        this.files = files;

        loadFile();
    }

    private void loadFile() throws IOException {
        System.out.println("Loading file " + files.get(this.fileNum));
        this.data.clear();
        BufferedReader in = new BufferedReader(new FileReader(new File(files.get(this.fileNum++))));
        String nextline = in.readLine();
        this.header = nextline;
        this.features = nextline.split(",").length - 1;
        while ((nextline = in.readLine()) != null) {
            String[] info = nextline.split(",");
            double[] dat = new double[info.length - 1];
            boolean valid = true;
            for (int i = 1; i < info.length; i++) {
                try {
                    dat[i - 1] = Double.parseDouble(info[i]);
                } catch (NumberFormatException e) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                this.data.add(dat);
            }
        }

        startOffsets.clear();
        int nMinibatchesPerEpoch = (data.size()-1) / exampleLength - 1;
        for(int i=-1; i<nMinibatchesPerEpoch; i++ ){
            startOffsets.add(i * exampleLength + (int)(Math.random() * exampleLength));
        }
        Collections.shuffle(startOffsets);
    }

    public boolean hasNext() {
        return fileNum != files.size()  || startOffsets.size() > 0;
	}

	public DataSet next() {
		return next(miniBatchSize);
	}

	public DataSet next(int num) {
		if (startOffsets.size() == 0) {
            try {
                loadFile();
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        }

        int currMinibatchSize = Math.min(num, startOffsets.size());

		//Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
		INDArray input = Nd4j.zeros(currMinibatchSize,features,exampleLength);
		INDArray labels = Nd4j.zeros(currMinibatchSize,features,exampleLength);

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = Math.max(0, startOffsets.removeFirst());
            int endIdx = Math.min(startIdx + exampleLength, data.size());
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                double[] give = data.get(j - 1);
                double[] pred = data.get(j);
                for (int k = 0; k < give.length; k++) {
                    input.putScalar(new int[]{i, k, c}, give[k]);
                    labels.putScalar(new int[]{i, k, c}, pred[k]);
                }
            }
        }
		return new DataSet(input,labels);
	}

	public int totalExamples() {
		return Integer.MAX_VALUE;
	}

	public int inputColumns() {
		return features;
	}

	public int totalOutcomes() {
		return features;
	}

	public void reset() {
        this.fileNum = 0;
        try {
            loadFile();
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Couldnt load next file in reset");
        }
    }

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return totalExamples();
	}

	public int numExamples() {
		return totalExamples();
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
