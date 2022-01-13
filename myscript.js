console.log('Hello TensorFlow');

var body = document.body;
import {MnistData} from './data.js';


// console.log(MnistData.testImages)
async function showExamples(data) {
  // Create a container in the visor
  const surface =
    tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  

  // Get the examples
  const examples = data.nextTestBatch(20);
//   const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];
  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    // console.log(imageTensor)
    
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style = 'margin: 4px;';
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();

  }
}

async function showData(data, target) {

  // Get the examples
  var number= data.targetTestBatch(1, target)

  const imageTensor = tf.tidy(() => {
    // Reshape the image to 28x28 px
    return number.xs
      .slice([0, 0], [1, number.xs.shape[1]])
      .reshape([28, 28, 1]);
  });

  
  const canvas = document.createElement('canvas');
  canvas.className='canvas';
  canvas.width = 28;
  canvas.height = 28;
  canvas.style = 'margin: 4px;';
  await tf.browser.toPixels(imageTensor, canvas);
  body.appendChild(canvas)

  imageTensor.dispose();
}

async function run() {
  const data = new MnistData();
  await data.load();
  await showExamples(data);
  console.log(data)

  const model = getModel();
  const encoder=getEncoderModel();
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  const predata = await train(model, encoder, data);

  const r = 1000
  const c = 2;
  const reshapeArray = (arr, r, c) => {
    if (r * c !== arr.length) {
        return arr
    }
    const res = []
    let row = []
    arr.forEach(item => {
        row.push(item*100)
        if (row.length === c) {
          res.push(row)
          row = []
        }
    })
    return res;
  };
  const newpre=reshapeArray(predata, r, c);

  console.log(newpre.length)
  var x_max=newpre[0][0];
  var x_min=newpre[0][0];
  var y_max=newpre[0][1];
  var y_min=newpre[0][1];
  // var current_x;
  // var current_y;
  for(var i=1; i < 1000; i++){
    var current_x = newpre[i][0];
    var current_y = newpre[i][1];
      if(current_x > x_max){
        x_max=current_x;
      }
      if(current_x < x_min){
        x_min=current_x;
      }
      if(current_y > y_max){
        y_max=current_y;
      }
      if(current_y < y_min){
        y_min=current_y;
      }
  }
  console.log(x_max, x_min, y_max, y_min)
  var width=(x_max-x_min).toFixed(1);
  var height=(y_max-y_min).toFixed(1);
  console.log(width, height)

  for(var num=0; num<newpre.length; num++){
    var x=newpre[num][0]
    var y=newpre[num][0]
    console.log(x)
    newpre[num][0]=(x-x_min).toFixed(1)
    newpre[num][1]=(y-y_min).toFixed(1)
  }

  let box=document.createElement('div')
  box.id='box';
  console.log("ok!!")
  for(var num=0; num<1000; num++){
    let current_num=num;
    let dot=document.createElement('div');
    dot.id=`dot${num}`;
    dot.className='dot';
    console.log(newpre[num][0])
    var X=(newpre[num][0]/width)*100;
    var Y=(newpre[num][1]/height)*100;
    dot.style.position="absolute";
    // dot.style.left="10px";
    // dot.style.bottom="10px";
    dot.style.left=`${X}%`;
    dot.style.bottom=`${Y}%`;
    dot.addEventListener("mouseover", ()=>{
      console.log("okkk")
      showData(data, current_num);
    })
    dot.addEventListener("mouseout", ()=>{
      const elements = document.getElementsByClassName('canvas');
      while(elements.length > 0){
          elements[0].parentNode.removeChild(elements[0]);
      }
    })
    box.appendChild(dot)
  }

  body.appendChild(box)
}



document.addEventListener('DOMContentLoaded', run);



function getModel() {
    const model = tf.sequential();
    
    // const IMAGE_WIDTH = 28;
    // const IMAGE_HEIGHT = 28;
    // const IMAGE_CHANNELS = 1;  

    // model.add(tf.layers.flatten({inputShape: 784}));
    console.log("ok1")

    // model.add(tf.layers.dense({
    //     inputShape: 784,
    //     units:784,
    //     activation: 'relu',
    // }));

    model.add(tf.layers.dense({
      inputShape: 784,
        units:150,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units:50,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units:10,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units:2,
    }));

    model.add(tf.layers.dense({
        units:10,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units:50,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units:150,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units:784,
        activation: 'relu',
    }));
  
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'meanSquaredError',
    });
  
    return model;
  }

  function getEncoderModel() {
    const en_model = tf.sequential();
    
    // const IMAGE_WIDTH = 28;
    // const IMAGE_HEIGHT = 28;
    // const IMAGE_CHANNELS = 1;  

    // model.add(tf.layers.flatten({inputShape: 784}));
    console.log("ok_encoder")

    en_model.add(tf.layers.dense({
      inputShape: 784,
        units:150,
        activation: 'relu',
    }));

    en_model.add(tf.layers.dense({
        units:50,
        activation: 'relu',
    }));

    en_model.add(tf.layers.dense({
        units:10,
        activation: 'relu',
    }));

    en_model.add(tf.layers.dense({
        units:2,
    }));
  
    // const optimizer = tf.train.adam();
    // en_model.compile({
    //   optimizer: optimizer,
    //   loss: 'categoricalCrossentropy',
    //   metrics: ['accuracy'],
    // });
  
    return en_model;
  }

  async function train(model, encoder, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = {
      name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;
  
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [
        d.xs.reshape([TRAIN_DATA_SIZE, 784]),
        d.labels
      ];
    });
  
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [
        d.xs.reshape([TEST_DATA_SIZE, 784]),
        d.labels
      ];
    });

    // console.log(testXs)
    // const tensor=testXs
    // const value = tensor.dataSync()
    // const arr=Array.from(value)
    // console.log(value[0])
    const history= await model.fit(trainXs, trainXs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testXs],
      epochs: 20,
      shuffle: true,
      callbacks: fitCallbacks
    });

    const pre=await encoder.predict(testXs)
    console.log(history)
    const predata=pre.dataSync()
    // return model.fit(trainXs, trainXs, {
    //   batchSize: BATCH_SIZE,
    //   validationData: [testXs, testXs],
    //   epochs: 20,
    //   shuffle: true,
    //   callbacks: fitCallbacks
    // });
    return predata;
    }


