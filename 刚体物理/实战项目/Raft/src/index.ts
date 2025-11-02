import * as fs from 'fs'
import * as path from 'path'

interface MetaData {
    vertexNum:number,
    edgesNumber:number,
    constraint:number,
}
interface DirectedConnection {
    start:number,
    end:number,
    cost:number
}

interface GraphData{
    metaData:MetaData,
    directedConnections:DirectedConnection[]
}


export class Solution{
    private graphData:GraphData
    private filename:string
    private parentNode: number[]
    private usedEdges:number[]
    private directedConnectedMapForSecondGraph:Map<number,Set<number>>
    private directedConnectedMapForFirstGraph:Map<number,Set<number>>
    constructor(filename:string){
        this.filename=filename;
        this.graphData=this.readGraphDataByFileName();
      
        this.parentNode=[];
        this.usedEdges=[];

        this.directedConnectedMapForSecondGraph=new Map<number,Set<number>>();
        this.getDirectedConnectedMap();

        this.directedConnectedMapForFirstGraph=new Map<number,Set<number>>();
    }

    public getResult=():number[]=>{
       return [this.minimumSpaningTreeForFirstGraph(),this.minimumSpaningTreeForSecondGraph()];
    }

    private minimumSpaningTreeForFirstGraph=():number=>{
        let cost=0;
        let curEdgeIdx=0;
        let usedEdgesNum=0;
        while (usedEdgesNum<this.graphData.metaData.vertexNum-1){
            const curEdge=this.graphData.directedConnections[curEdgeIdx] as DirectedConnection;
            const rootForStart=this.findRootNode(curEdge.start);
            const rootForEnd=this.findRootNode(curEdge.end);
            if(rootForStart!=rootForEnd){
                cost=(curEdge.cost%Math.pow(2,16)+cost)%Math.pow(2,16);
                this.usedEdges.push(curEdgeIdx);
                this.parentNode[Math.max(rootForStart,rootForEnd)]=Math.min(rootForStart,rootForEnd);
                if(!this.directedConnectedMapForFirstGraph.get(curEdge.start)) this.directedConnectedMapForFirstGraph.set(curEdge.start,new Set<number>());
                if(!this.directedConnectedMapForFirstGraph.get(curEdge.end)) this.directedConnectedMapForFirstGraph.set(curEdge.end,new Set<number>());
                this.directedConnectedMapForFirstGraph.get(curEdge.start)?.add(curEdge.end);
                this.directedConnectedMapForFirstGraph.get(curEdge.end)?.add(curEdge.start);
                usedEdgesNum++;
            }
            curEdgeIdx++;
        }
        return cost;
    }

    private minimumSpaningTreeForSecondGraph=():number=>{
        this.getFeasibleEdgesForAllNodes();
        let cost=0;
        let curEdgeIdx=0;
        let usedEdgesNum=0;
        this.parentNode=[];
        while (usedEdgesNum<this.graphData.metaData.vertexNum-1){
            const curEdge=this.graphData.directedConnections[curEdgeIdx] as DirectedConnection;
            if(this.usedEdges.indexOf(curEdgeIdx)==-1 && this.directedConnectedMapForSecondGraph.get(curEdge.start)?.has(curEdge.end)){
                const rootForStart=this.findRootNode(curEdge.start);
                const rootForEnd=this.findRootNode(curEdge.end);        
                if(rootForStart!=rootForEnd){
                    cost=(curEdge.cost%Math.pow(2,16)+cost)%Math.pow(2,16);
                    this.usedEdges.push(curEdgeIdx);
                    this.parentNode[Math.max(rootForStart,rootForEnd)]=Math.min(rootForStart,rootForEnd);
                    usedEdgesNum++;
                }
            }           
            curEdgeIdx++;
        }
        return cost;
    }

    private findRootNode=(nodeIdx:number):number=>{
        let curNodeIdx=nodeIdx;
        while(this.parentNode[curNodeIdx]!=undefined){
            curNodeIdx=this.parentNode[curNodeIdx] as number
        }
        return curNodeIdx;
    }

    private getFilePath=():string=>{
        const parentDir=path.dirname(__dirname);
        const filePath=path.join(parentDir,`./datapub/${this.filename}`);
        return filePath;
    }
    
    private readBuffer=():Buffer=>{
        const filePath=this.getFilePath();
        const buffer=fs.readFileSync(filePath);
        return buffer;
    }
    
    private readGraphDataByFileName=():GraphData=>{
        const buffer=this.readBuffer();
        const str=buffer.toString();
        const lines=str.split('\n');
        const directedConnections=[];
        const metaNum=lines[0]?.split(' ') as string[];
        const metaData:MetaData={
            vertexNum:Number(metaNum[0]),
            edgesNumber:Number(metaNum[1]),
            constraint:Number(metaNum[2]),
        }
        for(var i=1; i<lines.length; i++){
            const numbers=lines[i]?.split(' ') as string[];
            if(numbers.length<3) break;
            directedConnections.push({
                start:Number(numbers[0]),
                end:Number(numbers[1]) ,
                cost: Number(numbers[2])
            })
        }
        return {
            metaData:metaData,
            directedConnections:directedConnections
        };
    }

    private getDirectedConnectedMap=():void=>{
        for(var i=0; i<this.graphData.directedConnections.length; i++){
            const edge=this.graphData.directedConnections[i] as DirectedConnection
            if(!this.directedConnectedMapForSecondGraph.get(edge.start)) this.directedConnectedMapForSecondGraph.set(edge.start,new Set<number>());
            if(!this.directedConnectedMapForSecondGraph.get(edge.end)) this.directedConnectedMapForSecondGraph.set(edge.end,new Set<number>());
            this.directedConnectedMapForSecondGraph.get(edge.start)?.add(edge.end);
            this.directedConnectedMapForSecondGraph.get(edge.end)?.add(edge.start);
        }  
    }

    private getFeasibleEdgesForAllNodes=():void=>{
        for(var i=0; i<this.graphData.metaData.vertexNum; i++){
            const set1=this.directedConnectedMapForSecondGraph.get(i+1) as Set<number>;
            const set2=this.getAllReachAbleNodesForAGivenNode(i+1);
            const set1InArr=Array.from(set1);
            set1InArr.map((node)=>{
                if(!set2.has(node)) set1.delete(node);
            })
        }   
    }

    private getAllReachAbleNodesForAGivenNode=(node:number):Set<Number>=>{
        const constraint=this.graphData.metaData.constraint;
        const feasibleNodes=new Set<number>();
        feasibleNodes.add(node);
        for(var i=1; i<constraint; i++){
            const nodes=Array.from(feasibleNodes);
            nodes.map((node)=>{
                const connectedNodes=Array.from(this.directedConnectedMapForFirstGraph.get(node) as Set<number>);
                connectedNodes?.map((connectedNode)=>{
                    feasibleNodes.add(connectedNode);
                })
            })
        }
        return feasibleNodes
    }

}

for(var i=1; i<=10; i++){
    const num=i<10?'0'+i:i;
    const solution=new Solution(`pub${num}.in`);
    console.log(solution.getResult());
}










