#include "EverestRobotActor.h"
#include "EverestProtocol.h"
#include "Components/SceneComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Engine/StaticMesh.h"
#include "Materials/MaterialInterface.h"
#include "Dom/JsonObject.h"
#include "Misc/Paths.h"

AEverestRobotActor::AEverestRobotActor()
{
    PrimaryActorTick.bCanEverTick=true;
    Root=CreateDefaultSubobject<USceneComponent>(TEXT("RobotRoot")); SetRootComponent(Root);
}

USceneComponent* AEverestRobotActor::EnsureBody(const FString& Name)
{
    if (USceneComponent** Existing=BodyRoots.Find(Name)) return *Existing;
    USceneComponent* C=NewObject<USceneComponent>(this,*FString::Printf(TEXT("Body_%s"),*Name));
    C->SetupAttachment(Root); C->RegisterComponent(); BodyRoots.Add(Name,C); return C;
}

void AEverestRobotActor::ApplyScene(const TSharedPtr<FJsonObject>& Scene)
{
    if(!Scene.IsValid() || !Scene->HasField(TEXT("visuals"))) return;
    UMaterialInterface* RobotMat=LoadObject<UMaterialInterface>(nullptr,TEXT("/Game/Everest/Materials/M_G1Robot.M_G1Robot"));
    int32 VisualIndex=0;
    for(const TSharedPtr<FJsonValue>& Value: Scene->GetArrayField(TEXT("visuals")))
    {
        const TSharedPtr<FJsonObject> V=Value->AsObject(); if(!V.IsValid()) continue;
        const FString Body=V->GetStringField(TEXT("body")); const FString Url=V->GetStringField(TEXT("url"));
        const FString Base=FPaths::GetBaseFilename(Url); const FString Asset=FString::Printf(TEXT("/Game/Robots/G1/Meshes/%s.%s"),*Base,*Base);
        UStaticMesh* Mesh=LoadObject<UStaticMesh>(nullptr,*Asset); if(!Mesh) continue;
        UStaticMeshComponent* C=NewObject<UStaticMeshComponent>(this,*FString::Printf(TEXT("Visual_%s_%d"),*Base,VisualIndex++));
        C->SetStaticMesh(Mesh); C->SetCollisionEnabled(ECollisionEnabled::NoCollision); C->SetupAttachment(EnsureBody(Body)); C->RegisterComponent(); if(RobotMat) C->SetMaterial(0,RobotMat);
        C->SetRelativeLocation(EverestProtocol::ToUnrealPosition(V->GetArrayField(TEXT("position"))));
        C->SetRelativeRotation(EverestProtocol::ToUnrealQuat(V->GetArrayField(TEXT("quaternion"))));
        const auto& S=V->GetArrayField(TEXT("scale"));
        // SourceData/G1 is preconverted from metres to Unreal centimetres.
        C->SetRelativeScale3D(FVector(S[0]->AsNumber(),S[1]->AsNumber(),S[2]->AsNumber()));
    }
}

void AEverestRobotActor::ApplyFrame(const TSharedPtr<FJsonObject>& Frame)
{
    if(!Frame.IsValid()) return;
    const auto& Names=Frame->GetArrayField(TEXT("body_names")); const auto& Pos=Frame->GetArrayField(TEXT("body_pos_w")); const auto& Quat=Frame->GetArrayField(TEXT("body_quat_w"));
    const int32 N=FMath::Min3(Names.Num(),Pos.Num(),Quat.Num());
    for(int32 I=0;I<N;++I){ const FString Name=Names[I]->AsString(); const auto P=Pos[I]->AsArray(); const auto Q=Quat[I]->AsArray(); EnsureBody(Name); Targets.Add(Name,FTransform(EverestProtocol::ToUnrealQuat(Q),EverestProtocol::ToUnrealPosition(P))); }
}

void AEverestRobotActor::Tick(float Dt)
{
    Super::Tick(Dt);
    const float A=1.f-FMath::Exp(-Dt*22.f);
    for(const auto& It:Targets) if(USceneComponent** C=BodyRoots.Find(It.Key)){ const FTransform Cur=(*C)->GetComponentTransform(); FTransform B; B.Blend(Cur,It.Value,A); (*C)->SetWorldTransform(B); }
}

FVector AEverestRobotActor::GetPelvisLocation() const
{
    if(USceneComponent* const* P=BodyRoots.Find(TEXT("pelvis"))) return (*P)->GetComponentLocation();
    return GetActorLocation();
}
