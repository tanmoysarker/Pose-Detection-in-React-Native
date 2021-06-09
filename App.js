import React, { useState, useEffect } from "react";
import { ActivityIndicator, Text, View, ScrollView, StyleSheet, Button, Platform, Dimensions } from "react-native";
import Constants from "expo-constants";

// camera
import { Camera } from "expo-camera";

// tensorflow
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";

// canvas
import Canvas from "react-native-canvas";


export default function App() {

    //Tensorflow and Permissions
    const [posenetModel, setPosenetModel] = useState(null);
    const [frameworkReady, setFrameworkReady] = useState(false);
    const [loopStarted, setLoopStarted] = useState(false);
    const [pose, setPose] = useState(null);

    const TensorCamera = cameraWithTensors(Camera);
    let requestAnimationFrameId = 0;

    //performance hacks (Platform dependent)
    const textureDims = { width: 1600, height: 1200 };
    const tensorDims = { width: 200, height: 200 };

    const [ctx, setCanvasContext] = useState(null);

    const [debugText, setDebugText] = useState("");

    let cameraLoopStarted = false;

    //-----------------------------
    // Run effect once
    // 1. Check camera permissions
    // 2. Initialize TensorFlow
    // 3. Load Posenet Model
    //-----------------------------
    useEffect(() => {
        if (!frameworkReady) {
            (async () => {

                // check permissions
                const { status } = await Camera.requestPermissionsAsync();
                console.log(`permissions status: ${status}`);

                // we must always wait for the Tensorflow API to be ready before any TF operation...
                await tf.ready();
                console.log("TF is ready");

                // load the mobilenet model and save it in state
                setPosenetModel(await posenet.load({
                    architecture: "MobileNetV1",
                    outputStride: 16,
                    multiplier: 0.5,
                    quantBytes: 2
                }));
                console.log("Posenet model loaded");

                setFrameworkReady(true);
            })();
        }
    }, []);


    //--------------------------
    // Run onUnmount routine
    // for cancelling animation 
    // (if running) to avoid leaks
    //--------------------------
    useEffect(() => {
        return () => {
            console.log("Unmounted!");
            cancelAnimationFrame(requestAnimationFrameId);
        };
    }, [requestAnimationFrameId]);


    const getPrediction = async (tensor) => {
        if (!tensor || !posenetModel) return;

        const t0 = performance.now();
        // TENSORFLOW MAGIC HAPPENS HERE!
        const pose = await posenetModel.estimateSinglePose(tensor, 0.5, true, 16)     // cannot have async function within tf.tidy
        // console.log((performance.now() - t0));
        if (!pose) return;

        var numTensors = tf.memory().numTensors;
        drawSkeleton(pose);

        // posenet.load().then(function(net) {
        //     const pose = net.estimateSinglePose(imageElement, {
        //       flipHorizontal: true
        //     });
        //     return pose;
        //   }).then(function(pose){
        //     console.log(pose);
        console.log('check',drawSkeleton(pose))
    }

    
    const drawPoint = (x, y) => {
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
        ctx.closePath();
    }


    const drawSegment = (x1, y1, x2, y2) => {
        console.log('csdsdc>>>>',`${x1}, ${y1}, ${x2}, ${y2}`);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#00ff00";
        ctx.stroke();
        ctx.closePath();
    }

    //Here's where the matching should happen
    const drawSkeleton = (pose) => {
        console.log('pose>>>',pose);
       
        const minPartConfidence = 0.1;
        for (var i = 0; i < pose.keypoints.length; i++) {
            const keypoint = pose.keypoints[i];
            if (keypoint.score < minPartConfidence) {
                continue;
            }
            var setKeypoints1 = keypoint['position']['x']
            var setKeypoints2 = keypoint['position']['y']
            var setKeypoints = [setKeypoints1,setKeypoints2]
            console.log('keypoint>>>',setKeypoints);


            //Here's the position to match should be

            var handToFollow =  (
                48.62314365253374,
                158.82877097519454
            )
            if (pose !== null && handToFollow === setKeypoints){
                
                setPose(setKeypoints);
            }

            drawPoint(keypoint['position']['x'], keypoint['position']['y']);
        }
        const adjacentKeyPoints = posenet.getAdjacentKeyPoints(pose.keypoints, minPartConfidence);
        adjacentKeyPoints.forEach((keypoints) => {
            drawSegment(keypoints[0].position.x, keypoints[0].position.y, keypoints[1].position.x, keypoints[1].position.y);
        });
    }
    


    const handleCameraStream = (imageAsTensors) => {
        if (cameraLoopStarted) return;      // guarantees that the image loop only runs once
        cameraLoopStarted = true;
        const loop = async () => {
            if (frameworkReady) {
                const nextImageTensor = await imageAsTensors.next().value;
                await getPrediction(nextImageTensor);
                nextImageTensor.dispose();
            }
            requestAnimationFrameId = requestAnimationFrame(loop);
        };
        loop();
    }


    // https://js.tensorflow.org/api_react_native/0.2.1/#cameraWithTensors
    const renderCameraView = () => {
        return <View style={styles.cameraView}>
            <TensorCamera
                style={styles.camera}
                type={Camera.Constants.Type.front}
                zoom={0}
                cameraTextureHeight={textureDims.height}
                cameraTextureWidth={textureDims.width}
                resizeHeight={tensorDims.height}
                resizeWidth={tensorDims.width}
                resizeDepth={3}
                onReady={(imageAsTensors) => handleCameraStream(imageAsTensors)}
                autorender={true}
            />
        </View>;
    }


    const handleCanvas = (canvas) => {
        if (canvas === null) return;
        const ctx = canvas.getContext("2d");
        setCanvasContext(ctx);
    }


    return (
        <View style={styles.container}>
            <View style={styles.body}>
                {renderCameraView()}
                <Canvas ref={handleCanvas} style={styles.canvas} />
            </View>
            <Text>{debugText}</Text>
            {pose !== null ?
                <Text>Found at {pose}</Text>
            : <Text style={{paddingHorizontal: 30}}>Looking ...</Text>}
        </View>
    );
}

const CAM_WIDTH = Dimensions.get("window").width;
const CAM_HEIGHT = CAM_WIDTH * 4 / 3;

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: "flex-start",
        paddingTop: Constants.statusBarHeight,
        backgroundColor: "#E8E8E8"
    },
    body: {
    },
    cameraView: {
        width: CAM_WIDTH,
        height: CAM_HEIGHT
    },
    camera: {
        width: "100%",
        height: "100%",
        zIndex: 1
    },
    canvas: {
        width: CAM_WIDTH,
        height: CAM_HEIGHT,
        zIndex: 2,
        position: "absolute"
    }
});



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Stores the data; no need to worry about converting to strings ;)
// (key needs to be string tho)
const storeData = async (key, value) => {
    try {
        if (typeof value === "object") {
            value = "json|" + JSON.stringify(value);
        } else {
            value = typeof value + "|" + value;
        }
        await AsyncStorage.setItem(key, value);
    } catch (e) {
        // saving error
        console.log("storeData error: " + e.message);
    }
}

// Gets the data; no need to worry about converting from strings ;)
// (key needs to be string tho)
const getData = async (key) => {
    try {
        var value = await AsyncStorage.getItem(key);
        if (value !== null) {
            // value previously stored
            var type = value.split("|")[0];
            value = value.substr(type.length + 1);
            switch (type) {
                case "json":
                    value = JSON.parse(value);
                    break;
                case "boolean":
                    value = value === "true";
                    break;
                case "number":
                    value = Number(value);
                    break;
            }
            return value;
        }
    } catch (e) {
        // error reading value
        console.log("getData error: " + e.message);
    }
}