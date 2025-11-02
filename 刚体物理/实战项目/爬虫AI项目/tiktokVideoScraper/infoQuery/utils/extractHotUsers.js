module.exports=async function extractHotUsers(page){
    const getFollowersNumStr=`(${getFollowersNum.toString()})`;
    const hotUsers=await page.evaluate((getFollowersNumStr)=>{
        window.getFollowersNum=eval(getFollowersNumStr);
        const allUsers=document.querySelectorAll('a[data-e2e="search-user-info-container"]');
        const hotUsers=[];
        allUsers.forEach(item=>{
        const followersNum=getFollowersNum(item.querySelector('[data-e2e="search-follow-count"]').innerText);
            if(followersNum>10000){
                hotUsers.push({
                    userID:item.querySelector('[data-e2e="search-user-unique-id"]').innerText,
                    followersNum:getFollowersNum(item.querySelector('[data-e2e="search-follow-count"]').innerText)
                })
            }
        })
        hotUsers.sort((a,b)=>{
            return b.followersNum-a.followersNum
        })
        return hotUsers},
        getFollowersNumStr
    )
    return hotUsers;
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
    return result;
}