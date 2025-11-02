const logInExistence=require('./logInExistence')
const tackleLogIn=require('./tackleLogIn')
const delay=require('../../utils/delay')
module.exports=async function tackleLogInMain(page,maxRetryLimit){
    let count=0;
    let signInExistence=await logInExistence(page)
    while(signInExistence && count<maxRetryLimit){
        await tackleLogIn(page);
        count++
        await delay(500);
        signInExistence=await logInExistence(page);
        if(!signInExistence){
            console.log('登录框验证已破解')
        }else{
            console.log(`登录框验证第第${count}次失败`)
        }

    }
    if(signInExistence) console.log('登录框验证破解不成功')
    return signInExistence?false:true;
}