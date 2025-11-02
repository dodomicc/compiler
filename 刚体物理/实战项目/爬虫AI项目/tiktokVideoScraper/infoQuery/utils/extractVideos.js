module.exports=async function getAllLinkByUserId(page,userId){
    const allLinks=await page.evaluate(function(userId){
        const allLinks=[];
        const lists=document.querySelectorAll('.css-x6y88p-DivItemContainerV2')
        for(var i=0; i<lists.length; i++){
            try{
                const description=lists[i].querySelector('.css-1wrhn5c-AMetaCaptionLine');
                const likes=lists[i].querySelector('.css-dirst9-StrongVideoCount').innerHTML
                allLinks.push({
                    userId:userId,
                    href: description.href,
                    likesNum: likes,
                    description:description.title
                })
            }catch(err){
                continue;
            }
        }
            return allLinks
        },userId)
        return allLinks;
}

