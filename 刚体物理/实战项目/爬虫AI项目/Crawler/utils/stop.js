exports.stop=async function stop(waitTime){
    return new Promise(resolve=>{
        setTimeout(()=>{
            resolve(2);
        },waitTime)
    })
}