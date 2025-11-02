const Jimp = require('jimp');
module.exports=async function getSliderStart(src,width,height,sliderBarWidth){
    const image = await Jimp.read(src); 
    image.resize(width, height); 
    const rgbArray = new Array(height);
    for (let i = 0; i < height; i++)   rgbArray[i] = new Array(width);
    image.scan(0, 0, width, height, function(x, y, idx) {
        const red = this.bitmap.data[idx]*0.2126;
        const green = this.bitmap.data[idx + 1]* 0.7152;
        const blue = this.bitmap.data[idx + 2]* 0.0722;
        rgbArray[y][x] =red+green+blue>100?1:0;
    });
    let maxChangeCount = 0;
    let coordinateShift = 0;
    let result=[];
    for (let w = 1; w < width; w++) {
        let changeCount = 0;
        for (let h = 0; h < height; h++) {
            if (rgbArray[h][w]-rgbArray[h][w - 1] == -1) {
                changeCount++;
            }
        }
        result.push({
            width:w,
            changeCount:changeCount
        });
        if (changeCount > maxChangeCount) {
        maxChangeCount = changeCount;
        coordinateShift = w;
        }
    }
    result.sort((a,b)=>{return b.changeCount-a.changeCount})
    let left=result[0].width;
    let right=result[0].width;
    for(var i=1; i<result.length; i++){
        if(result[i].width>=left && result[i].width<=right){
            continue;
        }else if(result[i].width<left){
            if(right-result[i].width>=sliderBarWidth){
                break;
            }else{
                left=result[i].width;
            }
        }else{
            if(result[i].width-left>=sliderBarWidth){
                break;
            }else{
                right=result[i].width;
            }
        }
    }
    return left-6;
}