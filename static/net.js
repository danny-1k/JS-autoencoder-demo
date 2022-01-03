// Loading weights

// Definitely not the best way to load large arrays into memory
// in a BROWSER.
// sometimes it's like that :)

var weights = {
    'encoder': {},
    'decoder': {},
};

var biases = {
    'encoder': {},
    'decoder': {},
};

var xhr = new XMLHttpRequest();


for (sub of ['encoder', 'decoder']) {

    for (var i = 0; i < 9; i++) {
        if ((i % 2) == 0) {
            xhr.open('GET', `/get/w/${sub}/${i}`, async = false);
            xhr.send();
            weights[sub][i] = eval(xhr.responseText);


            xhr.open('GET', `/get/b/${sub}/${i}`, async = false);
            xhr.send();
            biases[sub][i] = eval(xhr.responseText);

        };


    };
};

// --------Utility functions--------//

function flatten(x) {
    // flatten n dimensional input
    // into 1d vector
    try {
        x = x.flat(Infinity);
    } catch {
        return [x];
    }
    return x;

}


function find_dims(arr) {
    // function to find the dimensions
    // of an array

    var dims = [];
    while (true) {
        try {
            dims.push(arr.length);
            arr = arr[0];
        } catch {
            break;
        };
    };
    var idx, el;
    for ([idx, el] of dims.entries()) {
        if (typeof (el) === "number") {
            continue;
        } else {
            dims.splice(idx, 1);
        };
    };
    return dims;

};

function reshape(arr, dims) {
    // function to reshape an
    // array

    var prod = 1;
    var arr_d_s = 1;
    find_dims(arr).forEach((i) => {
        arr_d_s = i * arr_d_s;
    });
    dims.forEach((i) => {
        prod = i * prod;
    });

    arr = flatten(arr);

    var new_arr = [];
    var idx, axis;
    for ([idx, axis] of dims.reverse().entries()) {
        var l = arr.length;
        var arr_copy = arr;

        for (var j = 0; j < (l / axis); j++) {
            var c = [];

            for (var i = 0; i < (axis); i++) {
                c.push(arr_copy[0]);
                arr_copy.shift();
            }

            new_arr.push(c);
            arr = new_arr;
        }
    }
    return new_arr[0];
}


// --------Vector & Matrix ops--------


function vector_mul(a, b) {
    // performs vector multiplication between two vectors
    // it is assumed that a and b have the same length

    var out = [];

    for (var i = 0; i < a.length; i++) {
        out.push(a[i] * b[i]);
    };

    return out;
};

function vector_add(a, b) {
    // performs vector addition between two vectors
    // it is assumed that a and b have the same length

    var out = [];

    for (var i = 0; i < a.length; i++) {
        out.push(a[i] + b[i]);
    };

    return out;
};

function scalar_add(a, b) {
    // Scalar addition

    var out = [];

    for (var i = 0; i < flatten(a).length; i++) {
        out.push(flatten(a)[i] + b);
    };

    return reshape(out, find_dims(a));
};

function scalar_mul(a, b) {
    // Scalar multiplication

    var out = [];

    for (var i = 0; i < flatten(a).length; i++) {
        out.push(flatten(a)[i] * b);
    };

    return reshape(out, find_dims(a));
};


function dot(a, b) {
    // performs dot product between two matrices

    if (find_dims(a)[1] !== find_dims(b)[0]) {
        console.log(`Warning!! a=>${find_dims(a)} b=>${find_dims(b)}`)
    };

    var out = [];

    for (var i = 0; i < a.length; i++) {
        var r = [];
        for (var j = 0; j < b[0].length; j++) {
            var vec = [];
            for (var k = 0; k < b.length; k++) {

                vec.push(b[k][j]);
            };

            r.push(sum(vector_mul(a[i], vec)));
        };

        out.push(r);
    };

    return out;
};

function sum(x) {
    // sum of a vector

    var sum = 0;

    for (var i = 0; i < x.length; i++) {
        sum += x[i];
    }

    return sum;
};


function max(x, b) {
    // maximum values between x and b

    var out = [];

    for (var i = 0; i < flatten(x).length; i++) {
        out.push(flatten(x)[i] >= b ? flatten(x)[i] : b);

    };

    return reshape(out, find_dims(x));


};


function min(x, b) {
    // minimum values between x and b

    var out = [];

    for (var i = 0; i < flatten(x).length; i++) {
        out.push(flatten(x)[i] < b ? flatten(x)[i] : b);

    };

    return reshape(out, find_dims(x));

};

function mul(x, b) {
    var out = [];

    for (var i = 0; i < flatten(x).length; i++) {
        out.push(flatten(x)[i] * b);
    };

    return reshape(out, find_dims(x));
};



// --------Neural net stuff--------



function linear(x, w, b) {
    return dot(x, w);
};


function add_bias(x, b) {
    out = [];

    for (var i = 0; i < x.length; i++) {
        out.push(scalar_add(x[i], b[i]));
    };

    return out;
};


function leaky_relu(x) {
    // f(x) = max(0,x) + alpha * min(0,x)
    out = vector_add(flatten(max(x, 0)), flatten(mul(min(x, 0), .01)));

    return reshape(out, find_dims(x));
};

function softmax(x) {
    // f(x) = e^x / sum(e^x)

    out = flatten(x);

    for (var i = 0; i < out.length; i++) {
        out[i] = Math.exp(out[i]);
    }

    out = scalar_mul(out, 1 / sum(out));

    return reshape(out, find_dims(x));
}


function encoder(x) {

    x = linear(x, weights['encoder'][0])
    x = add_bias(x, biases['encoder'][0])
    x = leaky_relu(x)

    x = linear(x, weights['encoder'][2])
    x = add_bias(x, biases['encoder'][2])
    x = leaky_relu(x)

    x = linear(x, weights['encoder'][4])
    x = add_bias(x, biases['encoder'][4])
    x = leaky_relu(x)

    x = linear(x, weights['encoder'][6])
    x = add_bias(x, biases['encoder'][6])
    x = leaky_relu(x)

    x = linear(x, weights['encoder'][8])
    x = add_bias(x, biases['encoder'][8])

    return x;
}

function decoder(x) {

    x = linear(x, weights['decoder'][0])
    x = add_bias(x, biases['decoder'][0])
    x = leaky_relu(x)

    x = linear(x, weights['decoder'][2])
    x = add_bias(x, biases['decoder'][2])
    x = leaky_relu(x)

    x = linear(x, weights['decoder'][4])
    x = add_bias(x, biases['decoder'][4])
    x = leaky_relu(x)

    x = linear(x, weights['decoder'][6])
    x = add_bias(x, biases['decoder'][6])
    x = leaky_relu(x)

    x = linear(x, weights['decoder'][8])
    x = add_bias(x, biases['decoder'][8])

    return x;

}



function autoencoder(x, normalize = false) {

    x = reshape(x, [1, 28 * 28])

    if (normalize) {

        var mean = .5
        var std = .5

        // Normalize 

        x = scalar_mul(x, 1 / max(x))

        // Standardize

        x = scalar_add(x, -mean) // subtract mean
        x = scalar_mul(x, 1 / std) // divide by std
    }


    x = encoder(x)
    x = decoder(x)

    return x;

}