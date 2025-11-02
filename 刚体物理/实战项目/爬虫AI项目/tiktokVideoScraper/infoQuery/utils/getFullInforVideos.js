const crypto=require('crypto')
module.exports=function getFullInfroForVideos(allLinks){
    const tagsRegex = /#(\S+)/g;
    const result=[];
    for(var i=0; i<allLinks.length; i++){
        if(allLinks[i].description.match(tagsRegex)!=null &&
        allLinks[i].description.match(tagsRegex)!=null){
            const content=allLinks[i].description.split('#')[0].trim();
            const tags=allLinks[i].description.match(tagsRegex).map((tag)=>{
                return tag.split('#')[1]
            });
            result.push({
                userId:allLinks[i].userId,
                videoId:crypto.createHash('md5').update(content+JSON.stringify(tags)).digest('hex'),
                href:allLinks[i].href,
                content:content,
                likesNum:getFollowersNum(allLinks[i].likesNum),
                tags:tags,
            })
        }
    }
    result.sort((a,b)=>{
        return b.likesNum-a.likesNum
    })
    return result;
}


function getFollowersNum(followersStr){
    const regex = /^[0-9]+(?:\.[0-9]+)?(?:K|M)?/;
    const numStr=followersStr.match(regex)[0];
    let result=0;
    if(numStr[numStr.length-1]=='K'||numStr[numStr.length-1]=='M'){
        result=Number.parseFloat(numStr.slice(0,numStr.length-1))
        result=numStr[numStr.length-1]=='K'?result*1000:result*1000*1000;
    }else{
        result= Number.parseInt(numStr);
    }
    return Math.floor(result);
}



