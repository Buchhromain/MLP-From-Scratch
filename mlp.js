const fs = require('fs');
const plotly = require('plotly')('username', 'apiKey'); // pas besoin si tu veux juste HTML
const path = require('path');

class MLP {
    constructor(config) {
        this.layers = [config.inputSize, ...config.hiddenSize, config.outputSize];
        this.learningRate = config.learningRate || 0.01;
        this.activation = (config.activation === "sigmoid") ? this.sigmoid : (config.activation === "tahn") ? this.tahn : this.relu;
        this.activationPrime = (config.activationPrime === "sigmoid") ? this.sigmoidPrime : (config.activationPrime === "tahn") ? this.tahnPrime : this.reluPrime;
        this.calculWeights = config.calculWeights;
        this.lossHistory = [];
        this.minDelta = config.minDelta || 0.000001;
        this.patiente = config.patiente || 10;
        this.initializeWeights();
        this.initializeBiases();
    }

    initializeWeights() {
        this.weights = [];
        for (let i = 0; i < this.layers.length - 1; i++) {
            if (this.calculWeights == "Xavier") {
                const fan_in = this.layers[i];
                const fan_out = this.layers[i + 1];
                const limit = Math.sqrt(6 / (fan_in + fan_out));
                const layerWeights = [];
                for (let j = 0; j < fan_in; j++) {
                    const neuronWeights = [];
                    for (let k = 0; k < fan_out; k++) {
                        neuronWeights.push(Math.random() * 2 * limit - limit); // Xavier Uniform
                    }
                    layerWeights.push(neuronWeights);
                }
                this.weights.push(layerWeights);
            } else {
                const layerWeights = [];
                for (let j = 0; j < this.layers[i]; j++) {
                    const neuronWeights = [];
                    for (let k = 0; k < this.layers[i + 1]; k++) {
                        neuronWeights.push(Math.random());
                    }
                    layerWeights.push(neuronWeights);
                }
                this.weights.push(layerWeights);
            }
        }
    }

    initializeBiases() {
        this.biases = new Array();
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.biases.push(Array.from({ length: this.layers[i + 1] }, () => Math.random()));
        }
    }

    sigmoid (z) {
        return 1 / (1 + Math.exp(-z));
    }

    tahn (z) {
        return Math.tanh(z);
    }

    relu (z) {
        return Math.max(0, z);
    }

    
    sigmoidPrime(z) {
        const sigmoidVal = this.activation(z);
        return sigmoidVal * (1 - sigmoidVal);
    }

    tahnPrime(z) {
        const tahnVal = this.activation(z);
        return 1 - Math.pow(tahnVal, 2);
    }

    reluPrime(z) {
        return z > 0 ? 1 : 0;
    }


    multiSigmoid (z) {
        let result = [];
        for (let i = 0; i < z.length; i++) {
            result.push(this.activation(z[i]));
        }
        return result;
    }

    forward(input) { // #1
        let activations = [input];
        let zs = [];
        for (let layers = 0; layers < this.layers.length - 1; layers++) {
            let weights = this.weights[layers];
            let z = this.dotProduct(input, weights);
            let biases = this.biases[layers];
            for (let i = 0; i < z.length; i++) {
                z[i] += biases[i];
            }
            zs.push(z);
            z = this.multiSigmoid(z);
            activations.push(z);
            input = z;
        }
        return {activations, zs};
    }


    loss(output, target) { // Mean Squared Error (calcule l'erreur quadratique moyenne)
        let sum = 0;
        for (let i = 0; i < output.length; i++) {
            sum += Math.pow(output[i] - target[i], 2);
        }
        return sum / output.length;
    }

    dotProduct(a, b) {
        let result = [];
        for (let i = 0; i < b[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < a.length; j++) {
                sum += a[j] * b[j][i]; // Multiplie chaque élément du vecteur par la colonne correspondante dans la matrice
            }
            result.push(sum);
        }
        return result;
    }

    dotProduct2(weights, delta) {
        let result = [];
        for (let i = 0; i < weights.length; i++) {
            let sum = 0;
            for (let j = 0; j < weights[i].length; j++) { 
                sum += weights[i][j] * delta[j];
            }
            result.push(sum);
        }
        return result;
    }

    delta(output, target, z) {
        let delta = [];
        for (let i = 0; i < output.length; i++) {
            delta.push((output[i] - target[i]) * this.sigmoidPrime(z[i]));
        }
        return delta;
    }

    deltaHidden(delta, weights, z) {
        const dot = this.dotProduct2(weights, delta);
        return dot.map((value, i) => value * this.sigmoidPrime(z[i]));
    }
    

    backpropagation(target, forwardOutput) { // #2
        let activations = forwardOutput.activations;
        let zs = forwardOutput.zs;

        let deltas = [];
        let output = activations[activations.length - 1];
        let z = zs[zs.length - 1];
        let delta = this.delta(output, target, z);
        deltas.push(delta);
        for (let i = this.layers.length - 2 ; i >= 1; i--) {
            let w = this.weights[i];
            z = zs[i - 1];
            delta = this.deltaHidden(delta, w, z);
            deltas.push(delta);
        }
        // console.log("Deltas:", deltas);

        deltas.reverse();

        let gradientsWeights = [];
        let gradientsBiases = [];

        for (let i = 0; i < deltas.length; i++) {
            let delta = deltas[i];
            let activationPrev = activations[i];
        
            let gradientWeight = [];
            for (let a = 0; a < activationPrev.length; a++) {
                let row = [];
                for (let d = 0; d < delta.length; d++) {
                    row.push(activationPrev[a] * delta[d]);
                }
                gradientWeight.push(row);
            }
        
            gradientsWeights.push(gradientWeight);
            gradientsBiases.push(delta);
        }

        for (let i = 0; i < this.layers.length - 1; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    this.weights[i][j][k] -= this.learningRate * gradientsWeights[i][j][k];
                }
            }
            for (let j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] -= this.learningRate * gradientsBiases[i][j];
            }
        }
        return {
            gradientsWeights,
            gradientsBiases
        };
    }

    train(data, config = {}) {
        let epochs = config.epochs || 10000;
        let betloss = Infinity
        let wait = 0
        for (let i = 0; i < epochs; i++) {
            let totalLoss = 0;
            for (let j = 0; j < data.length; j++) {
                let input = data[j].input;
                let target = data[j].target;
                let forwardOutput = monMLP.forward(input);
                let loss = monMLP.loss(forwardOutput.activations.at(-1), target) 
                monMLP.backpropagation(target, forwardOutput);
                totalLoss += loss
            }
            this.lossHistory.push(totalLoss / data.length);
            if (i % 100 === 0) { // Affiche tous les 100 epochs par exemple
                console.log(`Epoch ${i} - Total Loss: ${totalLoss}`);
            }
            if (betloss - totalLoss > this.minDelta) {
                betloss = totalLoss;
                wait = 0;
            } else {
                wait ++;
                if (wait > this.patiente) {
                    console.log("Arret de l'apprentissage");
                    break;
                }
            }
        }
    }

    predict(data) {
        console.log("Prediction :");
        for (let d of data) {
            let out = monMLP.forward(d.input).activations.pop();
            let rounded = out.map(x => Math.round(x * 10) / 1); // arrondi à 2 chiffres après la virgule
            console.log(`Input: ${d.input} => Output: ${out} (arrondi: ${rounded})`);
        }
    }

    plotLossHistory(lossHistory, filename = "loss_plot.html") {
        const trace = {
            x: Array.from({ length: lossHistory.length }, (_, i) => i),
            y: lossHistory,
            type: 'scatter',
            mode: 'lines',
            name: 'Loss',
            line: { color: 'blue' }
        };
    
        const layout = {
            title: 'Loss Evolution',
            xaxis: { title: 'Epochs' },
            yaxis: { title: 'Loss' }
        };
    
        const htmlContent = `
            <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div id="plot"></div>
                <script>
                    var data = ${JSON.stringify([trace])};
                    var layout = ${JSON.stringify(layout)};
                    Plotly.newPlot('plot', data, layout);
                </script>
            </body>
            </html>
        `;
    
        fs.writeFileSync(path.join(__dirname, filename), htmlContent);
        console.log(`Graphique sauvegardé dans ${filename}`);
    }
    
}

let data = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [0] },
    { input: [1, 0], target: [0] },
    { input: [1, 1], target: [0.1] },
    { input: [2, 1], target: [0.2] },
    { input: [1, 2], target: [0.2] },
    { input: [2, 2], target: [0.4] },
    { input: [3, 3], target: [0.9] },
    { input: [4, 2], target: [0.8] },
];

monMLP = new MLP({
    inputSize: 2,
    hiddenSize: [4, 8],
    outputSize: 1,
    learningRate: 0.1,
    activation: "sigmoid",
    activationPrime: "sigmoidPrime",
    calculWeights: "Xavier",
    patiente: 1000, //doesn't work well
    minDelta: 0.0001 //doesn't work well
})
monMLP.train(data, {epochs: 100000});
monMLP.plotLossHistory(monMLP.lossHistory);
monMLP.predict(data);
