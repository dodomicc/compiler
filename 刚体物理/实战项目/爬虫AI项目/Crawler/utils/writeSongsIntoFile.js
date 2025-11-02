const fs=require('fs')
exports.writeSongsIntoFile=function(songs,ListInfo,i){
    if(!fs.existsSync(`./songs/${ListInfo[i].name}`))
    fs.mkdirSync(`./songs/${ListInfo[i].name}`,{recursive:true})
    const str=JSON.stringify(songs);
    let writeText='';
    for(var j=0; j<str.length; j++){
        if(j%140==0) writeText+='\n'
        writeText+=str[j]
    }
    if(fs.existsSync(`./songs/${ListInfo[i].name}/${ListInfo[i].name}.txt`))
    fs.truncateSync(`./songs/${ListInfo[i].name}/${ListInfo[i].name}.txt`, 0, (err) => {
        if (err) {
            console.error('清空文件失败:', err);
        } else {
            console.log('文件已成功清空');
        }
    });
    fs.writeFileSync(`./songs/${ListInfo[i].name}/${ListInfo[i].name}.txt`,writeText,{flag:'a'})
}