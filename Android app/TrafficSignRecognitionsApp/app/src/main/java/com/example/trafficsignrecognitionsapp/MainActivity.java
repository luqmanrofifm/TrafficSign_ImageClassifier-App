package com.example.trafficsignrecognitionsapp;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.os.Trace;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private static final int RC_IC_PIC = 101;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int MAX_RESULTS = 1;

    private ImageView imgpenampil;
    private Button btnkamera;
    private TextView resultanswer,motivasi;
    private Bitmap gambarawal;
    private String modelName = "model.tflite";
    private String labelName = "labels.txt";
    private MappedByteBuffer tfliteModel;
    private Interpreter interpreter;
    private List<String> labels;
    private ByteBuffer byteBuffer;
    private int numThreads = 1;
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private final int[] intValues = new int[224*224];
    private float[][] labelProbArray = null;
    private List<Recognition> result = null;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imgpenampil = findViewById(R.id.img_penampil);
        btnkamera = findViewById(R.id.btn_kamera);
        resultanswer = findViewById(R.id.resultrext);
        motivasi = findViewById(R.id.quote);
        btnkamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode==RC_IC_PIC){
            if(resultCode==RESULT_OK){
                this.gambarawal = (Bitmap) data.getExtras().get("data");
                imgpenampil.getLayoutParams().height = 500;
                imgpenampil.setScaleType(ImageView.ScaleType.FIT_CENTER);
                imgpenampil.setImageBitmap(gambarawal);
                try {
                    this.tfliteModel = loadTFLITEModel(this);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                try {
                    this.labels = loadLabel(this);
                } catch (IOException e) {
                    e.printStackTrace();
                    Toast.makeText(this, "Dibatalkan", Toast.LENGTH_SHORT).show();
                }
                processingImage(gambarawal);

            }

        }

        else if(resultCode==RESULT_CANCELED){
            Toast.makeText(this, "Dibatalkan", Toast.LENGTH_SHORT).show();
        }
    }


    private void processingImage(Bitmap bitmap){
        tfliteOptions.setNumThreads(numThreads);
        interpreter = new Interpreter(tfliteModel,tfliteOptions);
        Bitmap gambar = ThumbnailUtils.extractThumbnail(bitmap, 224,224);
        convertBitmapToBuffer(gambar);
        labelProbArray = new float[1][labels.size()];
        interpreter.run(byteBuffer,labelProbArray);
        result = getSorted(labelProbArray);
        if(result != null && result.size() >= 1){
            Recognition recognition = result.get(0);
            if(recognition != null){
                if(recognition.getTitle() != null)
                    resultanswer.setText(recognition.getTitle());
            }
        }
        else {
            resultanswer.setText("Tidak ada rambu yang cocok");
        }
        String kata = "\"Mari patuhi rambu lalu lintas agar selamat sampai tujuan\"";
        motivasi.setText(kata);
    }

    private List<Recognition> getSorted(float[][] labelProbArray) {
        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < labels.size(); ++i) {
            pq.add(
                    new Recognition(
                            "" + i,
                            labels.size() > i ? labels.get(i) : "unknown",
                            labelProbArray[0][i],
                            null));
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

    private void convertBitmapToBuffer(Bitmap bitmap) {
        byteBuffer = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE
                *224*224
                *DIM_PIXEL_SIZE*4
        );
        byteBuffer.order(ByteOrder.nativeOrder());
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel = 0;
        for(int i=0; i < 224; ++i){
            for(int j=0; j < 224; ++j){
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)- 127.5f)/127.5f);
                byteBuffer.putFloat((((val >> 16) & 8)- 127.5f)/127.5f);
                byteBuffer.putFloat(((val & 0xFF)- 127.5f)/127.5f);
            }
        }
    }

    private List<String> loadLabel(Activity activity) throws IOException {
        List<String>label = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(labelName)));
        String line;
        while((line = reader.readLine())!= null){
            label.add(line);
        }
        reader.close();
        return label;
    }

    private MappedByteBuffer loadTFLITEModel(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void openCamera(){
        Intent kameraintent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(kameraintent, RC_IC_PIC);
    }

}
