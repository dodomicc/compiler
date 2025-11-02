const {tiktokdl}=require('tiktokdl');
module.exports=async function getVideoOriginalURL(href,maxTryTime){
    let count=0;
    let result={}
    console.log('---------------开始获取视频下载的原始url--------------------')
    count=1;
    let downloadURL = await getDownloadURL(href,count)
    while(count<maxTryTime && !downloadURL){
        count++
        downloadURL = await getDownloadURL(href,count)
    }
    if(downloadURL){
        console.log('---------------视频下载的原始url获取成功--------------------')
        result={
            musicURL:downloadURL.music,
            videoURL:downloadURL.video,
        };
        
        return result;
    }else{
        console.log('---------------视频下载的原始url获取失败--------------------')
        
        throw new Error('视频下载的原始url获取失败')
        
    }
  
}



async function getDownloadURL(href,count){
    try{
        const downloadVideoInfo = await tiktokdl(href)
        return downloadVideoInfo;
    }catch(err){
        return;
    }
}